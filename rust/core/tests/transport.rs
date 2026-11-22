use std::io::{Cursor, Read};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyllm_core::errors::ErrorKind;
use anyllm_core::transport::{
    iter_sse, iter_sse_controlled, join_url, open_stream_controlled, raise_for_status,
    request_json, user_agent, MAX_RETRIES,
};
use serde_json::json;

struct Bytewise<R>(R);

impl<R: Read> Read for Bytewise<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        self.0.read(&mut buf[..1])
    }
}

#[test]
fn sse_splits_events_at_every_chunk_boundary_and_accepts_mixed_line_endings() {
    let body = b"data: first\r\n\ndata: second\n\r\ndata: third\r\n\r\ndata: [DONE]\r\n\r\n";
    let events: Vec<String> = iter_sse(Bytewise(Cursor::new(body.to_vec())))
        .map(|event| event.expect("valid SSE event"))
        .collect();
    assert_eq!(events, ["first", "second", "third"]);
}

#[test]
fn sse_flushes_trailing_event_only_at_end_of_stream() {
    let events: Vec<String> = iter_sse(Cursor::new(b"data: final".to_vec()))
        .map(|event| event.expect("valid SSE event"))
        .collect();
    assert_eq!(events, ["final"]);
}

struct Disconnect;

impl Read for Disconnect {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "peer disconnected",
        ))
    }
}

#[test]
fn sse_disconnect_becomes_network_error() {
    let error = iter_sse(Disconnect).next().unwrap().unwrap_err();
    assert_eq!(error.kind, ErrorKind::Network);
    assert!(error.message.contains("peer disconnected"));
    assert!(error.hint.unwrap().contains("provider may have timed out"));
}

struct GateReader(Arc<(Mutex<bool>, Condvar)>);

impl Read for GateReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let (open, changed) = &*self.0;
        let mut open = open.lock().unwrap();
        while !*open {
            open = changed.wait(open).unwrap();
        }
        *open = false;
        let bytes = b"data: resumed\n\n";
        let amount = bytes.len().min(buf.len());
        buf[..amount].copy_from_slice(&bytes[..amount]);
        Ok(amount)
    }
}

#[test]
fn controlled_sse_honors_deadline_while_reader_is_blocked() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let mut events = iter_sse_controlled(
        GateReader(Arc::clone(&gate)),
        Some(Instant::now() + Duration::from_millis(30)),
        Arc::new(AtomicBool::new(false)),
    );
    let error = events.next().expect("deadline error").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Network);
    assert!(error.message.contains("overall deadline elapsed"));

    let (open, changed) = &*gate;
    *open.lock().unwrap() = true;
    changed.notify_all();
}

#[test]
fn controlled_sse_honors_cancellation_flag() {
    let cancelled = Arc::new(AtomicBool::new(true));
    let mut events =
        iter_sse_controlled(Cursor::new(b"data: ignored\n\n".to_vec()), None, cancelled);
    let error = events.next().expect("cancel error").unwrap_err();
    assert!(error.message.contains("stopped locally after a cancel"));
}

#[test]
fn url_joining_and_user_agent_are_well_formed() {
    assert_eq!(
        join_url("https://example.test/v1?api-version=1", "/models"),
        "https://example.test/v1/models?api-version=1"
    );
    let agent = user_agent();
    assert!(agent.starts_with("anyllm/"));
    assert!(agent.contains("rust"));
    assert_eq!(MAX_RETRIES, 2);
}

#[test]
fn status_errors_keep_status_param_code_and_retry_after_hint() {
    let error = raise_for_status(
        429,
        br#"{"error":{"message":"slow down","param":"model","code":"quota"}}"#,
        Some("4"),
        "https://example.test/v1",
    );
    assert_eq!(error.kind, ErrorKind::RateLimit);
    assert!(error.message.contains("slow down (model)"));
    assert!(error.hint.unwrap().contains("4 seconds"));

    let error = raise_for_status(
        422,
        br#"{"error":{"message":"bad payload","param":"input","code":"invalid"}}"#,
        None,
        "https://example.test/v1",
    );
    assert_eq!(error.kind, ErrorKind::Api);
    assert_eq!(error.status, Some(422));
    assert_eq!(error.param.as_deref(), Some("input"));
    assert_eq!(error.code.as_deref(), Some("invalid"));
}

fn read_request(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut request = Vec::new();
    let mut byte = [0; 1];
    while !request.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap() == 0 {
            break;
        }
        request.push(byte[0]);
    }
    String::from_utf8_lossy(&request).to_string()
}

fn respond(stream: &mut TcpStream, status: &str, headers: &str, body: &str) {
    use std::io::Write;
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

#[test]
fn json_request_sets_user_agent_encodes_json_and_retries_get_503() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        for (status, body) in [
            ("503 Service Unavailable", "{}"),
            ("200 OK", "{\"ok\":true}"),
        ] {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_request(&mut stream);
            assert!(request.to_ascii_lowercase().contains("user-agent: anyllm/"));
            respond(
                &mut stream,
                status,
                if status.starts_with("503") {
                    "Retry-After: 0\r\n"
                } else {
                    ""
                },
                body,
            );
        }
    });

    let value = request_json(
        &format!("http://{address}/models"),
        &[],
        None,
        "GET",
        2.0,
        1,
    )
    .unwrap();
    assert_eq!(value, json!({"ok": true}));
    server.join().unwrap();
}

#[test]
fn json_post_serializes_utf8_and_does_not_retry_ambiguous_server_error() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_request(&mut stream);
        assert!(request.starts_with("POST "));
        assert!(request
            .to_ascii_lowercase()
            .contains("content-type: application/json"));
        let content_length = request
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length: ")
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap();
        let mut body = vec![0; content_length];
        stream.read_exact(&mut body).unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            json!({"name":"café"})
        );
        respond(
            &mut stream,
            "500 Internal Server Error",
            "",
            "{\"error\":\"failed\"}",
        );
        assert!(listener.set_nonblocking(true).is_ok());
        assert!(
            listener.accept().is_err(),
            "POST must not be replayed after a 500"
        );
    });

    let error = request_json(
        &format!("http://{address}/completion"),
        &[],
        Some(&json!({"name":"café"})),
        "POST",
        2.0,
        2,
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Api);
    assert_eq!(error.status, Some(500));
    server.join().unwrap();
}

#[test]
fn json_request_applies_read_timeout_before_status_line() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = read_request(&mut stream);
        thread::sleep(Duration::from_millis(250));
        let _ = respond(&mut stream, "200 OK", "", "{}");
    });

    let started = Instant::now();
    let error =
        request_json(&format!("http://{address}/slow"), &[], None, "GET", 0.1, 0).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Network);
    assert!(started.elapsed() < Duration::from_millis(220));
    server.join().unwrap();
}

#[test]
fn controlled_open_checks_cancellation_before_opening_request() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let cancelled = Arc::new(AtomicBool::new(true));
    let error = open_stream_controlled(
        &format!("http://{address}/stream"),
        &[],
        &json!({"stream":true}),
        1.0,
        2,
        None,
        Some(cancelled),
    )
    .err()
    .expect("cancelled request must fail");
    assert!(error.message.contains("cancel"));
    assert!(listener.accept().is_err(), "cancelled request was sent");
}

#[test]
fn controlled_open_interrupts_retry_delay_on_cancellation() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let cancelled = Arc::new(AtomicBool::new(false));
    let server_cancelled = Arc::clone(&cancelled);
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = read_request(&mut stream);
        respond(
            &mut stream,
            "503 Service Unavailable",
            "Retry-After: 2\r\n",
            "{}",
        );
        thread::sleep(Duration::from_millis(20));
        server_cancelled.store(true, Ordering::Relaxed);
        listener.set_nonblocking(true).unwrap();
        assert!(listener.accept().is_err(), "cancelled retry was sent");
    });

    let started = Instant::now();
    let error = open_stream_controlled(
        &format!("http://{address}/stream"),
        &[],
        &json!({"stream":true}),
        2.0,
        2,
        None,
        Some(cancelled),
    )
    .err()
    .expect("cancelled retry must fail");
    assert!(error.message.contains("cancel"));
    assert!(started.elapsed() < Duration::from_millis(500));
    server.join().unwrap();
}

#[test]
fn controlled_open_deadline_interrupts_retry_wait() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = read_request(&mut stream);
        respond(
            &mut stream,
            "503 Service Unavailable",
            "Retry-After: 2\r\n",
            "{}",
        );
        listener.set_nonblocking(true).unwrap();
        assert!(listener.accept().is_err(), "request ran past its deadline");
    });

    let started = Instant::now();
    let error = open_stream_controlled(
        &format!("http://{address}/stream"),
        &[],
        &json!({"stream":true}),
        2.0,
        2,
        Some(started + Duration::from_millis(100)),
        None,
    )
    .err()
    .expect("deadline must stop retry");
    assert!(error.message.contains("deadline"));
    assert!(started.elapsed() < Duration::from_millis(500));
    server.join().unwrap();
}

#[test]
fn controlled_open_deadline_bounds_a_blocked_request_attempt() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = read_request(&mut stream);
        thread::sleep(Duration::from_millis(250));
        respond(&mut stream, "200 OK", "", "{}");
    });

    let started = Instant::now();
    let error = open_stream_controlled(
        &format!("http://{address}/stream"),
        &[],
        &json!({"stream":true}),
        2.0,
        0,
        Some(started + Duration::from_millis(100)),
        None,
    )
    .err()
    .expect("deadline must stop the request attempt");
    assert!(error.message.contains("deadline"));
    assert!(started.elapsed() < Duration::from_millis(220));
    server.join().unwrap();
}
