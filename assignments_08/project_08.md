## Supabase set up confirmation

Project is setup correctly, the tables were created and contain the corresponding columns, Row Level Security was also disabled following the instructions.

## Summary of estimates

Scenario A costs $1.66 per month for part-time use, whereas Scenario B totals $2,579.65 per month due to a 24/7 GPU instance, managed database, and a Terabyte of S3 storage. The surprissing gap isn't surprising given hardware costs, but it highlights how quickly continuous uptime and active databases inflate cloud expenses. 

Beyond these scenarios, the calculator shows deep contrast in pricing structures: keeping a relational database or compute node alive 24/7 quickly drains budgets, whereas raw object storage like S3 remains remarkably inexpensive at scale.

The vast cost difference underscores that a GPU instance is strictly a high-value investment meant for heavy parallel processing workloads where massive speedups justify the high hourly rate; for standard or idle workloads, it's a misuse of capital.

## Video link

https://youtu.be/qrtTU5ChNhM