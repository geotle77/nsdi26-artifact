# Mock Data Schema (Open-Source Template)

This document defines the minimal mock input files for open-source usage when private production data is unavailable.

## Supported mock files

- `data/host_fault_detail.example.json`
- `data/normal_hosts.example.json` (optional)

To use these with commands from `README.md`, copy them to the expected runtime names:

```bash
cp data/host_fault_detail.example.json data/host_fault_detail.json
cp data/normal_hosts.example.json data/normal_hosts.json
```

## 1) `host_fault_detail` schema

Top-level type: `object`

- Key: host identifier string (for example, `node-0001.example.org`)
- Value: array of fault event objects for that host

### Fault event object fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `FaultID` | string | Yes | Mock fault ID. Keep non-sensitive (for example, `MOCK-FAULT-0001`). |
| `FaultType` | string | Optional | High-level category (for example, `Network`, `Power`, `Storage`). |
| `SubClass` | string | Yes | Detailed fault subclass used for labeling and feature generation. |
| `Class` | string | Yes | Fault class group (for example, `Network`, `Storage`, `Other`). |
| `Level` | string | Yes | Severity/group label (for example, `Hardware Fault`, `Other Fault`). |
| `CreatedTime` | integer | Yes | Event start timestamp in Unix milliseconds. |
| `CreatedTimeStr` | string | Recommended | Human-readable UTC/local time string (`YYYY-MM-DD HH:MM:SS`). |
| `ClosedTime` | integer or `null` | Yes | Event end timestamp in Unix milliseconds, or `null` if unresolved. |
| `ClosedTimeStr` | string | Recommended | End time string or empty string when unresolved. |

### Minimal example structure

```json
{
  "node-0001.example.org": [
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

Each item is a host identifier with no fault records in the selected data period.

Example:

```json
[
  "node-0101.example.org",
  "node-0102.example.org"
]
```

## Privacy and open-source safety

- Use placeholder hostnames only (`*.example.org` is recommended).
- Do not include hardware model identifiers, serial numbers, rack positions, internal region codes, or asset tags.
- Do not include private network addresses, account IDs, or production ticket IDs.
- Keep all IDs and labels synthetic and reproducible.
