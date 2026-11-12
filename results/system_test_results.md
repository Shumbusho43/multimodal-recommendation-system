# System Testing Results

Run: 2026-11-12T14:58:27  |  Mode: REAL models

| # | Scenario | Face file | Face result | Voice file | Voice result | Expected | Actual | Result |
|---|----------|-----------|-------------|------------|--------------|----------|--------|--------|
| 1 | Authorized transaction (Allan: face + voice) | Allan_neutral.jpg | allan (similarity 1.000) | Allan_confirm.wav | allan (confidence 0.975) | GRANTED | GRANTED | PASS |
| 2 | Authorized transaction (Nazira: face + voice, HEIC photo) | Nazira_neutral.HEIC | nazira (similarity 1.000) | Nazira_approve.wav | nazira (confidence 0.970) | GRANTED | GRANTED | PASS |
| 3 | Unauthorized attempt: stranger's face photo | unknown_astronaut.png | rejected (similarity 0.216) | Allan_confirm.wav | not run | DENIED_FACE | DENIED_FACE | PASS |
| 4 | Unauthorized attempt: valid face, unknown voice | Allan_neutral.jpg | allan (similarity 1.000) | unknown_confirm.wav | unknown (confidence 0.475) | DENIED_VOICE | DENIED_VOICE | PASS |
| 5 | Identity mismatch: Allan's face, David's voice | Allan_neutral.jpg | allan (similarity 1.000) | David_approve.wav | david (confidence 0.965) | DENIED_VOICE | DENIED_VOICE | PASS |
