//! Conversion between the core provider implementation and orchestration APIs.
use anyllm_core as core;
use anyllm_orch::sys;
use std::collections::BTreeMap;

pub fn provider(p: &core::Provider) -> sys::Provider {
    sys::Provider {
        name: p.name.clone(),
        label: p.label.clone(),
        kind: p.kind.clone(),
        base_url: p.base_url.clone(),
        env_vars: p.env_vars.clone(),
        models: p.models.clone(),
        key_required: p.key_required,
        local: p.local,
        docs: p.docs.clone(),
        auth_header: p.auth_header.clone(),
        auth_prefix: p.auth_prefix.clone(),
    }
}
pub fn core_provider(p: &sys::Provider) -> core::Provider {
    core::Provider {
        name: p.name.clone(),
        label: p.label.clone(),
        kind: p.kind.clone(),
        base_url: p.base_url.clone(),
        env_vars: p.env_vars.clone(),
        models: p.models.clone(),
        key_required: p.key_required,
        local: p.local,
        docs: p.docs.clone(),
        auth_header: p.auth_header.clone(),
        auth_prefix: p.auth_prefix.clone(),
    }
}
pub fn providers(p: &BTreeMap<String, core::Provider>) -> sys::ProviderMap {
    p.iter().map(|(k, v)| (k.clone(), provider(v))).collect()
}
pub fn error(e: core::AnyLLMError) -> sys::Error {
    let kind = match e.kind {
        core::ErrorKind::Usage => sys::ErrorKind::Usage,
        core::ErrorKind::Config => sys::ErrorKind::Config,
        core::ErrorKind::Auth => sys::ErrorKind::Auth,
        core::ErrorKind::NotFound => sys::ErrorKind::NotFound,
        core::ErrorKind::RateLimit => sys::ErrorKind::RateLimit,
        core::ErrorKind::Network => sys::ErrorKind::Network,
        core::ErrorKind::Api => sys::ErrorKind::Api,
        core::ErrorKind::Provider => sys::ErrorKind::Provider,
    };
    sys::Error {
        kind,
        message: e.message,
        hint: e.hint,
        status: e.status,
        param: e.param,
        code: e.code,
    }
}
pub fn core_error(e: sys::Error) -> core::AnyLLMError {
    let kind = match e.kind {
        sys::ErrorKind::Usage => core::ErrorKind::Usage,
        sys::ErrorKind::Config => core::ErrorKind::Config,
        sys::ErrorKind::Auth => core::ErrorKind::Auth,
        sys::ErrorKind::NotFound => core::ErrorKind::NotFound,
        sys::ErrorKind::RateLimit => core::ErrorKind::RateLimit,
        sys::ErrorKind::Network => core::ErrorKind::Network,
        sys::ErrorKind::Api => core::ErrorKind::Api,
        sys::ErrorKind::Provider => core::ErrorKind::Provider,
    };
    core::AnyLLMError {
        kind,
        message: e.message,
        hint: e.hint,
        status: e.status,
        param: e.param,
        code: e.code,
    }
}
pub struct Store(pub core::KeyStore);
impl sys::KeyStore for Store {
    fn get(&self, name: &str) -> sys::Result<Option<String>> {
        self.0.get(name).map_err(error)
    }
}
pub struct Backend;
impl sys::ChatBackend for Backend {
    fn stream(&self, request: &sys::ChatRequest) -> sys::Result<Vec<sys::Chunk>> {
        let p = core_provider(&request.provider);
        let messages: Vec<core::Message> = request
            .messages
            .iter()
            .map(|m| core::Message {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();
        let args = core::StreamArgs {
            model: &request.model,
            messages: &messages,
            system: request.system.as_deref(),
            key: request.credential.as_deref(),
            base_url: Some(&request.base_url),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            timeout: request.timeout,
            tools: request.tools.as_ref(),
            deadline: request.deadline,
            cancel: request.cancel.clone(),
            ..Default::default()
        };
        let mut stream = core::stream_chat(&p, &args).map_err(error)?;
        let mut chunks = Vec::new();
        while let Some(c) = stream.next_chunk().map_err(error)? {
            chunks.push(sys::Chunk {
                text: c.text,
                reasoning: c.reasoning,
                usage: c.usage.map(|u| sys::Usage {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                    reasoning_tokens: u.reasoning_tokens,
                    cached_tokens: u.cached_tokens,
                }),
                finish_reason: c.finish_reason,
                tool_call: c.tool_call.map(|t| sys::ToolCallDelta {
                    index: t.index,
                    id: t.id,
                    name: t.name,
                    arguments: t.arguments,
                }),
            });
        }
        Ok(chunks)
    }
}

/// Host for orchestration slash-command handlers. Synchronization updates only
/// core-owned fields, keeping orchestration-only state in the host session.
pub struct ReplHost {
    pub style: sys::Style,
    pub providers: sys::ProviderMap,
    pub store: Store,
    pub session: sys::Session,
}
impl ReplHost {
    pub fn from_core(
        session: &core::Session,
        style: core::Style,
        providers: &BTreeMap<String, core::Provider>,
        store: &core::KeyStore,
    ) -> Self {
        let mut state = sys::Session::new(provider(&session.provider), &session.model);
        state.base_url = session.base_url.clone();
        state.api_key = session.api_key.clone();
        state.key_source = session.key_source.clone();
        state.system = session.system.clone();
        state.temperature = session.temperature;
        state.max_tokens = session.max_tokens;
        state.timeout = session.timeout;
        state.show_reasoning = session.show_reasoning;
        state.target_alias = session.target_alias.clone();
        state.messages = session
            .messages
            .iter()
            .map(|m| sys::Message {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();
        state.usage = sys::Usage {
            input_tokens: session.usage.input_tokens,
            output_tokens: session.usage.output_tokens,
            reasoning_tokens: session.usage.reasoning_tokens,
            cached_tokens: session.usage.cached_tokens,
        };
        state.turns = session.turns as u64;
        Self {
            style: sys::Style::new(style.enabled),
            providers: self::providers(providers),
            store: Store(core::KeyStore {
                path: store.path.clone(),
            }),
            session: state,
        }
    }
    pub fn sync_core(&self, session: &mut core::Session) {
        let s = &self.session;
        session.provider = core_provider(&s.provider);
        session.model = s.model.clone();
        session.base_url = s.base_url.clone();
        session.api_key = s.api_key.clone();
        session.key_source = s.key_source.clone();
        session.system = s.system.clone();
        session.temperature = s.temperature;
        session.max_tokens = s.max_tokens;
        session.timeout = s.timeout;
        session.show_reasoning = s.show_reasoning;
        session.target_alias = s.target_alias.clone();
        session.messages = s
            .messages
            .iter()
            .map(|m| core::Message {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();
        session.usage = core::Usage {
            input_tokens: s.usage.input_tokens,
            output_tokens: s.usage.output_tokens,
            reasoning_tokens: s.usage.reasoning_tokens,
            cached_tokens: s.usage.cached_tokens,
        };
        session.turns = s.turns as usize;
    }
}
impl anyllm_orch::seam::Repl for ReplHost {
    fn say(&mut self, text: &str) {
        println!("{}", text);
    }
    fn style(&self) -> &sys::Style {
        &self.style
    }
    fn providers(&self) -> &sys::ProviderMap {
        &self.providers
    }
    fn store(&self) -> &dyn sys::KeyStore {
        &self.store
    }
    fn session(&mut self) -> &mut sys::Session {
        &mut self.session
    }
    fn is_interactive(&self) -> bool {
        use std::io::IsTerminal;
        std::io::stdin().is_terminal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyllm_orch::sys::ChatBackend;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{atomic::AtomicBool, Arc};
    #[test]
    fn cancelled_request_does_not_connect() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let mut provider = sys::Provider::new("mock", "Mock", "openai", &url);
        provider.key_required = false;
        let request = sys::ChatRequest {
            provider,
            model: "m".into(),
            base_url: url,
            messages: vec![sys::Message::new("user", "hello")],
            system: None,
            credential: None,
            temperature: None,
            max_tokens: None,
            timeout: 1.0,
            tools: None,
            deadline: None,
            cancel: Some(Arc::new(AtomicBool::new(true))),
        };
        assert!(Backend.stream(&request).is_err());
        assert!(listener.accept().is_err());
    }
    #[test]
    fn forwards_tools_messages_and_stream_deltas() {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = server.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut socket, _) = server.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut data = Vec::new();
            let mut buf = [0; 4096];
            loop {
                let n = socket.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                data.extend_from_slice(&buf[..n]);
                if let Some(end) = data.windows(4).position(|x| x == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&data[..end]);
                    let length = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|x| x.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if data.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let text = String::from_utf8(data).unwrap();
            let body = text.split("\r\n\r\n").nth(1).unwrap();
            let body: serde_json::Value = serde_json::from_str(body).unwrap();
            assert_eq!(body["messages"][1]["tool_call_id"], "call-1");
            assert_eq!(body["tools"][0]["type"], "function");
            let s="data: {\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",s.len(),s).unwrap();
        });
        let mut provider =
            sys::Provider::new("mock", "Mock", "openai", &format!("http://{address}/v1"));
        provider.key_required = false;
        let request = sys::ChatRequest {
            provider,
            model: "model".into(),
            base_url: format!("http://{address}/v1"),
            messages: vec![
                sys::Message::new("user", "hi"),
                sys::Message {
                    role: "tool".into(),
                    content: "result".into(),
                    tool_calls: None,
                    tool_call_id: Some("call-1".into()),
                },
            ],
            system: None,
            credential: None,
            temperature: None,
            max_tokens: None,
            timeout: 5.0,
            tools: Some(
                serde_json::json!([{"type":"function","function":{"name":"read_attached_file"}}]),
            ),
            deadline: None,
            cancel: Some(Arc::new(AtomicBool::new(false))),
        };
        let chunks = Backend.stream(&request).unwrap();
        assert_eq!(chunks[0].text, "hello");
        handle.join().unwrap();
    }
}
