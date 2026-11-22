use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use anyllm_core::{stream_chat, Message, Provider, StreamArgs, ANTHROPIC, GEMINI, OPENAI};
use serde_json::{json, Value};

fn read_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buf = [0; 4096];
    loop {
        let n = stream.read(&mut buf).unwrap_or(0);
        if n == 0 {
            break;
        }
        bytes.extend_from_slice(&buf[..n]);
        if let Some(pos) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            let head_end = pos + 4;
            let head = String::from_utf8_lossy(&bytes[..head_end]).to_string();
            let length = head
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            while bytes.len() < head_end + length {
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break;
                }
                bytes.extend_from_slice(&buf[..n]);
            }
            let body = bytes[head_end..head_end + length.min(bytes.len() - head_end)].to_vec();
            return (head, body);
        }
    }
    (String::from_utf8_lossy(&bytes).to_string(), Vec::new())
}

fn serve(sse: &'static str) -> (String, thread::JoinHandle<(String, Value)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let (head, body) = read_request(&mut stream);
        let request = serde_json::from_slice(&body).unwrap_or(Value::Null);
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            sse.len(),
            sse
        )
        .unwrap();
        (head, request)
    });
    (format!("http://{address}/v1"), handle)
}

fn collect(provider: &Provider, model: &str) -> (String, String, Option<String>) {
    let messages = [Message::new("user", "hello")];
    let args = StreamArgs {
        model,
        messages: &messages,
        key: Some("test-secret"),
        timeout: 2.0,
        ..Default::default()
    };
    let mut stream = stream_chat(provider, &args).unwrap();
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut finish = None;
    while let Some(chunk) = stream.next_chunk().unwrap() {
        text.push_str(&chunk.text);
        reasoning.push_str(&chunk.reasoning);
        finish = chunk.finish_reason.or(finish);
    }
    (text, reasoning, finish)
}

#[test]
fn openai_mock_streams_text_usage_and_finish() {
    let (base, server) = serve("data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n");
    let provider = Provider::new(
        "mock",
        "Mock",
        OPENAI,
        &base,
        &[],
        &[],
        true,
        false,
        "",
        "Authorization",
        "Bearer ",
    );
    let result = collect(&provider, "mock-model");
    assert_eq!(
        result,
        ("Hello world".into(), String::new(), Some("stop".into()))
    );
    let (head, request) = server.join().unwrap();
    assert!(head.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(head
        .to_ascii_lowercase()
        .contains("authorization: bearer test-secret"));
    assert_eq!(request["stream_options"]["include_usage"], json!(true));
    assert_eq!(request["messages"][0]["content"], json!("hello"));
}

#[test]
fn anthropic_mock_streams_text_and_end_turn() {
    let (base, server) = serve("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi from Claude\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    let provider = Provider::new(
        "mock",
        "Mock",
        ANTHROPIC,
        &base,
        &[],
        &[],
        true,
        false,
        "",
        "x-api-key",
        "",
    );
    let result = collect(&provider, "claude-mock");
    assert_eq!(result.0, "Hi from Claude");
    assert_eq!(result.2.as_deref(), Some("end_turn"));
    let (head, request) = server.join().unwrap();
    assert!(head.starts_with("POST /v1/messages HTTP/1.1"));
    assert!(head.to_ascii_lowercase().contains("x-api-key: test-secret"));
    assert_eq!(request["messages"][0]["role"], json!("user"));
}

#[test]
fn gemini_mock_streams_text_and_finish_reason() {
    let (base, server) = serve("data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello Gemini\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2}}\n\n");
    let provider = Provider::new(
        "google",
        "Google Gemini",
        GEMINI,
        &base,
        &[],
        &[],
        true,
        false,
        "",
        "x-goog-api-key",
        "",
    );
    let result = collect(&provider, "gemini-mock");
    assert_eq!(result.0, "Hello Gemini");
    assert_eq!(result.2.as_deref(), Some("stop"));
    let (head, request) = server.join().unwrap();
    assert!(head.starts_with("POST /v1/models/gemini-mock:streamGenerateContent?alt=sse HTTP/1.1"));
    assert!(head
        .to_ascii_lowercase()
        .contains("x-goog-api-key: test-secret"));
    assert_eq!(request["contents"][0]["parts"][0]["text"], json!("hello"));
}
