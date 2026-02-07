# Mock Data Schema

This document defines the mock input contracts used by the open-source workflow.

## Tracked templates

- `data/host_fault_detail.example.json`
- `data/normal_hosts.example.json` (optional)

## Runtime files

You may copy templates to runtime names:

```bash
cp data/host_fault_detail.example.json data/host_fault_detail.json
cp data/normal_hosts.example.json data/normal_hosts.json
```

The generator also accepts custom paths through CLI flags.

## 1) `host_fault_detail` schema

Top-level type: `object`

- Key: host identifier string, for example `host-0001.example.com`
- Value: array of fault events for that host

### Fault event fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `FaultID` | string | Yes | Mock fault identifier such as `MOCK-FAULT-0001`. |
| `FaultType` | string | Optional | High-level category such as `Network` or `Storage`. |
| `SubClass` | string | Yes | Fine-grained fault category. |
| `Class` | string | Yes | Mid-level category. |
| `Level` | string | Yes | Severity label. |
| `CreatedTime` | integer | Yes | Start timestamp in Unix milliseconds. |
| `CreatedTimeStr` | string | Recommended | Human-readable timestamp (`YYYY-MM-DD HH:MM:SS`). |
| `ClosedTime` | integer or `null` | Yes | End timestamp in Unix milliseconds. |
| `ClosedTimeStr` | string | Recommended | End timestamp string or empty string when unresolved. |

### Minimal example

```json
{
  "host-0001.example.com": [
    {
      "FaultID": "MOCK-FAULT-0001",
      "FaultType": "Network",
      "SubClass": "Link Down",
      "Class": "Network",
      "Level": "Hardware Fault",
      "CreatedTime": 1735689600000,
      "CreatedTimeStr": "2025-01-01 00:00:00",
      "ClosedTime": 1735693200000,
      "ClosedTimeStr": "2025-01-01 01:00:00"
    }
  ]
}
```

## 2) `normal_hosts` schema (optional)

Top-level type: `array[string]`

Each item is a host with no fault records in the selected period.

## 3) `all_hosts_info` schema (optional)

Top-level type: `object`

- Key: host identifier string
- Value: metadata object used as optional categorical features

Supported fields:

- `model` (string)
- `gpu_model` (string)
- `cpu_model` (string)
- `quota_group` (string)
- `manufacturer` (string)
- `main_board.model` (string)

If this file is not provided, the pipeline fills metadata fields with `Unknown`.

## Safety rules

- Use synthetic identifiers only.
- Never commit production records.
- Keep all host names and IDs anonymized.
