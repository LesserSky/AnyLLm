use anyllm_orch::seam::{make_snapshot, ApprovedAttachment};
use anyllm_orch::sys::ToolCallDelta;
use anyllm_orch::tool_protocol::{
    assemble_tool_calls, execute_read_attached_file, openai_tool_payload, parse_path_argument,
    AssembledCall, ARGUMENT_BYTE_CAP, READ_ATTACHED_FILE, RESULT_BYTE_CAP,
};
use serde_json::json;

#[test]
fn assembles_interleaved_fragments_in_first_seen_order() {
    let deltas = vec![
        ToolCallDelta {
            index: 1,
            id: "b".into(),
            name: "read_".into(),
            arguments: "{\"path\":".into(),
        },
        ToolCallDelta {
            index: 0,
            id: "a".into(),
            name: "read_attached_file".into(),
            arguments: "{\"path\":\"x\"}".into(),
        },
        ToolCallDelta {
            index: 1,
            name: "attached_file".into(),
            arguments: "\"y\"}".into(),
            ..Default::default()
        },
    ];
    let calls = assemble_tool_calls(&deltas).unwrap();
    assert_eq!(
        calls[0],
        AssembledCall {
            index: 1,
            id: "b".into(),
            name: "read_attached_file".into(),
            arguments: "{\"path\":\"y\"}".into()
        }
    );
    assert_eq!(calls[1].id, "a");
}

#[test]
fn rejects_invalid_or_oversized_tool_arguments() {
    assert_eq!(
        parse_path_argument(Some(r#"{"path":"a b.txt"}"#)).unwrap(),
        "a b.txt"
    );
    for bad in [
        None,
        Some("[]"),
        Some(r#"{"path":"x","other":1}"#),
        Some(r#"{"path":""}"#),
    ] {
        assert!(parse_path_argument(bad).is_err());
    }
    assert!(parse_path_argument(Some(&"x".repeat(ARGUMENT_BYTE_CAP + 1))).is_err());
    let bad_id = vec![
        ToolCallDelta {
            index: 0,
            id: "a".into(),
            ..Default::default()
        },
        ToolCallDelta {
            index: 0,
            id: "b".into(),
            ..Default::default()
        },
    ];
    assert!(assemble_tool_calls(&bad_id).is_err());
}

#[test]
fn payload_contains_only_the_attachment_reader() {
    let payload = openai_tool_payload(&[READ_ATTACHED_FILE.into()]).unwrap();
    assert_eq!(payload[0]["function"]["name"], READ_ATTACHED_FILE);
    assert!(payload[0]["function"]["parameters"]["additionalProperties"] == false);
    assert!(openai_tool_payload(&["shell".into()]).is_err());
}

#[test]
fn only_approved_attachment_text_is_returned_and_truncated_by_bytes() {
    let approved = ApprovedAttachment::new("docs/a.txt", "hash", "hello");
    let snapshot = make_snapshot(
        &[],
        None,
        None,
        None,
        Some(vec![approved]),
        false,
        Some("test"),
    );
    let allowed = execute_read_attached_file("docs/a.txt", &snapshot);
    assert_eq!(allowed["ok"], true);
    assert_eq!(allowed["content"], "hello");
    assert_eq!(
        execute_read_attached_file("/etc/passwd", &snapshot)["ok"],
        false
    );

    let large = ApprovedAttachment::new("large", "hash", &"x".repeat(RESULT_BYTE_CAP + 10));
    let snapshot = make_snapshot(
        &[],
        None,
        None,
        None,
        Some(vec![large]),
        false,
        Some("test2"),
    );
    let result = execute_read_attached_file("large", &snapshot);
    assert_eq!(result["truncated"], true);
    assert_eq!(result["content"].as_str().unwrap().len(), RESULT_BYTE_CAP);
    let _ = json!(result);
}
