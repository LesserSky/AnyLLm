use std::io::{Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyllm_core::adapters::int_field;
use anyllm_core::catalog::{Provider, ANTHROPIC, GEMINI, OPENAI};
use anyllm_core::{list_models, stream_chat, ChunkStream, Message, StreamArgs};
use serde_json::{json, Value};

struct Request {
    method: String,
    target: String,
    headers: String,
    body: Vec<u8>,
}

fn receive_request(stream: &mut TcpStream) -> Request {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    let (header_end, content_length) = loop {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0, "client closed before sending request");
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let header = String::from_utf8_lossy(&bytes[..end]);
            let content_length = header
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            break (end, content_length);
        }
    };
    while bytes.len() < header_end + 4 + content_length {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0, "client closed before request body was complete");
        bytes.extend_from_slice(&buffer[..count]);
    }

    let headers = String::from_utf8_lossy(&bytes[..header_end]).to_string();
    let mut request_line = headers.lines().next().unwrap().split_whitespace();
    let method = request_line.next().unwrap().to_string();
    let target = request_line.next().unwrap().to_string();
    let body_start = header_end + 4;
    Request {
        method,
        target,
        headers,
        body: bytes[body_start..body_start + content_length].to_vec(),
    }
}

fn serve_once(body: &'static str, content_type: &'static str) -> (String, JoinHandle<Request>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let worker = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let request = receive_request(&mut stream);
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        request
    });
    (format!("http://{address}/v1"), worker)
}

fn provider(name: &str, label: &str, kind: &str, base_url: &str) -> Provider {
    Provider::new(
        name,
        label,
        kind,
        base_url,
        &[],
        &[],
        true,
        false,
        "",
        "x-api-key",
        "Token ",
    )
}

#[test]
fn openai_compatible_provider_sends_payload_and_parses_tool_deltas() {
    let (base, server) = serve_once(
        concat!(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
            "\n\ndata: [DONE]\n\n",
        ),
        "text/event-stream",
    );
    let custom = provider("custom", "Custom", OPENAI, &base);
    let messages = [Message::new("user", "find x")];
    let args = StreamArgs {
        model: "custom-model",
        messages: &messages,
        system: Some("Be precise"),
        key: Some("secret"),
        temperature: Some(0.25),
        max_tokens: Some(80),
        ..StreamArgs::default()
    };
    let mut stream = stream_chat(&custom, &args).unwrap();
    let tool = stream.next_chunk().unwrap().unwrap().tool_call.unwrap();
    assert_eq!(tool.index, 0);
    assert_eq!(tool.id, "call_1");
    assert_eq!(tool.name, "lookup");
    assert_eq!(tool.arguments, "{}");
    assert_eq!(
        stream
            .next_chunk()
            .unwrap()
            .unwrap()
            .finish_reason
            .as_deref(),
        Some("tool_calls")
    );

    let request = server.join().unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.target, "/v1/chat/completions");
    assert!(request
        .headers
        .to_ascii_lowercase()
        .contains("x-api-key: token secret"));
    let payload: Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(payload["model"], "custom-model");
    assert_eq!(
        payload["messages"][0],
        json!({"role":"system", "content":"Be precise"})
    );
    assert_eq!(
        payload["messages"][1],
        json!({"role":"user", "content":"find x"})
    );
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["temperature"], 0.25);
    assert_eq!(payload["max_tokens"], 80);
    assert_eq!(payload["stream_options"]["include_usage"], true);
}

#[test]
fn model_listing_uses_provider_routes_auth_and_generation_filter() {
    let (base, server) = serve_once(
        r#"{"data":[{"id":"z-model"},{"id":"A-model"},{"id":"a-model"},{"name":"models/x-model"}]}"#,
        "application/json",
    );
    let custom = provider("compat", "Compat", OPENAI, &base);
    assert_eq!(
        list_models(&custom, Some("key"), None, 2.0).unwrap(),
        ["A-model", "a-model", "x-model", "z-model"]
    );
    let request = server.join().unwrap();
    assert_eq!(request.method, "GET");
    assert_eq!(request.target, "/v1/models");
    assert!(request
        .headers
        .to_ascii_lowercase()
        .contains("x-api-key: token key"));

    let (base, server) = serve_once(r#"{"data":[{"id":"claude-test"}]}"#, "application/json");
    let anthropic = provider("anthropic", "Anthropic", ANTHROPIC, &base);
    assert_eq!(
        list_models(&anthropic, Some("key"), None, 2.0).unwrap(),
        ["claude-test"]
    );
    let request = server.join().unwrap();
    assert_eq!(request.target, "/v1/models?limit=1000");
    assert!(request
        .headers
        .to_ascii_lowercase()
        .contains("anthropic-version: 2023-06-01"));

    let (base, server) = serve_once(
        r#"{"models":[{"name":"models/gemini-z","supportedGenerationMethods":["generateContent"]},{"name":"models/embed","supportedGenerationMethods":["embedContent"]},{"name":"models/gemini-a","supportedGenerationMethods":[]},{"name":"models/gemini-b"}]}"#,
        "application/json",
    );
    let gemini = provider("google", "Gemini", GEMINI, &base);
    assert_eq!(
        list_models(&gemini, Some("key"), None, 2.0).unwrap(),
        ["gemini-a", "gemini-b", "gemini-z"]
    );
    let request = server.join().unwrap();
    assert_eq!(request.target, "/v1/models?pageSize=1000");
    assert!(request
        .headers
        .to_ascii_lowercase()
        .contains("x-api-key: token key"));
}

#[test]
fn empty_base_url_falls_back_to_provider_default() {
    let (base, server) = serve_once(r#"{"data":[{"id":"model"}]}"#, "application/json");
    let custom = provider("compat", "Compat", OPENAI, &base);
    assert_eq!(
        list_models(&custom, None, Some(""), 2.0).unwrap(),
        ["model"]
    );
    assert_eq!(server.join().unwrap().target, "/v1/models");
}

#[test]
fn integer_coercion_matches_python_for_strings_and_numbers() {
    assert_eq!(int_field(Some(&json!(2.9))), Some(2));
    assert_eq!(int_field(Some(&json!("29"))), Some(29));
    assert_eq!(int_field(Some(&json!("2.9"))), None);
    assert_eq!(int_field(Some(&json!("-2"))), None);
    assert_eq!(int_field(Some(&json!(true))), None);
}

#[test]
fn provider_error_messages_do_not_include_json_string_quotes() {
    let body = b"data: {\"error\":{\"message\":\"invalid request\"}}\n\n";
    let mut stream = ChunkStream::from_reader(OPENAI, "Mock", Box::new(Cursor::new(body.to_vec())));
    let error = stream.next_chunk().unwrap_err();
    assert_eq!(error.message, "Mock error: invalid request");
}

#[test]
fn zero_token_limit_uses_each_providers_python_default() {
    let (base, server) = serve_once("data: [DONE]\n\n", "text/event-stream");
    let openai = provider("compat", "Compat", OPENAI, &base);
    let args = StreamArgs {
        model: "gpt-4o-mini",
        key: Some("key"),
        max_tokens: Some(0),
        ..StreamArgs::default()
    };
    drop(stream_chat(&openai, &args).unwrap());
    let request = server.join().unwrap();
    let payload: Value = serde_json::from_slice(&request.body).unwrap();
    assert!(payload.get("max_tokens").is_none());

    let (base, server) = serve_once("data: {\"type\":\"message_stop\"}\n\n", "text/event-stream");
    let anthropic = provider("anthropic", "Anthropic", ANTHROPIC, &base);
    let args = StreamArgs {
        model: "claude-test",
        key: Some("key"),
        max_tokens: Some(0),
        ..StreamArgs::default()
    };
    drop(stream_chat(&anthropic, &args).unwrap());
    let request = server.join().unwrap();
    let payload: Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(payload["max_tokens"], 4096);

    let (base, server) = serve_once("data: {}\n\n", "text/event-stream");
    let gemini = provider("google", "Gemini", GEMINI, &base);
    let args = StreamArgs {
        model: "gemini-test",
        key: Some("key"),
        max_tokens: Some(0),
        ..StreamArgs::default()
    };
    drop(stream_chat(&gemini, &args).unwrap());
    let request = server.join().unwrap();
    let payload: Value = serde_json::from_slice(&request.body).unwrap();
    assert!(payload.get("generationConfig").is_none());
}
