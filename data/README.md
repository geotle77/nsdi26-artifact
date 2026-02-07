# Data Directory Policy

This repository only tracks mock example files.

Do not commit production or sensitive data.
Do not include private hostnames, IP addresses, account IDs, serial numbers, asset tags, or internal ticket identifiers.

Use these templates:

- `data/host_fault_detail.example.json`
- `data/normal_hosts.example.json`

Before running the pipeline, copy templates to runtime filenames:

```bash
cp data/host_fault_detail.example.json data/host_fault_detail.json
cp data/normal_hosts.example.json data/normal_hosts.json
```

You can also provide custom paths through CLI arguments.
