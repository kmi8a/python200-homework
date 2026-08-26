# Part 1: Warmup — Cloud Concepts

## Cloud Concepts Question 1

### What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The economic model of cloud computing is 'pay-as-you-go', and it differs from owning your own servers in the sense that is more flexible because you dont need an initial investment in hardware, and also you dont need to buy more hardware to scale up the operations.

## Cloud Concepts Question 2

### What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling refers to upgrading the machines to be more powerful (memory, computing power, storage, etc), while horizontal scaling refers to adding more machines, generally less powerful, that can split the load between them.

a service that process video in high resolution would be a good example thaht might benefit more from vertical scaling, while a service that handles viral content would be a good example of horizontal scaling

## For the three scenarios below, write one sentence saying which type of scaling applies and why.

### A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

Horizontal scaling applies, as the massive surge in traffic can be distributed across multiple independent server instances behind a load balancer.

### A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

Vertical scaling applies, as deep learning frameworks require a single powerful machine with an upgraded GPU and more RAM to fit and train complex models in memory.

### A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

Horizontal scaling applies, since the batch of 10,000 files can be easily partitioned and processed simultaneously across multiple worker nodes.

## Cloud Concepts Question 3

Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

Gmail:
(SaaS) It provides a fully functional email application directly to end users. 

Azure Virtual Machines:
(IaaS) Provides raw, configurable virtual servers where users manage the operating system, networking, and storage.

AWS S3 (Simple Storage Service):
(IaaS) Offers scalable cloud storage infrastructure accessible via APIs as a foundational building block for applications.

GitHub Codespaces:
(PaaS) It provides a pre-configured cloud development environment with all the necessary tools and runtimes built-in.

Snowflake:
(SaaS) delivers a fully managed cloud data platform where users can query and analyze data without maintaining database servers.

Supabase:
(PaaS) It supplies pre-built backend features like databases, authentication, and APIs so developers can skip building server-side logic from scratch.

### Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

IaaS means that the cloud provider is responsible for the hardware and its maintenance, and the user is responsible for everything else. and example of this is Azure Virtual Machines, were you just rent a node on the cloud but you are responsible for the OS, and any applications that need to run for your particular project

PaaS means that the cloud provider is responsible for the hardware and also all of the underlying runtime, servers and scaling, while the user is responsible for the custom code an example of this is Github Codespaces, where the user is provided with a pre-configured cloud development environment with all the necessary tools and runtimes built-in

SaaS means that the user is the end user of the application, on this case the provider is responsible for everything, from the underlying infrastructure, servers, and databases all the way to the application code, security patches, and user interface, examples of this are Gmail, or any POS system.

## Cloud Concepts Question 4

### What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

Platforms like DataBricks or Snowflake offer a curated version of the same services offered by the traditional cloud providers but instead they wrap everything into a cohesive, highly optimized SaaS experience, offering an abstraction layer that deals with the setting up, configuring and managing of these services.

With the use of these platforms you gain developer velocity, automated elasticity and automated management, and you give up some financial cost as you pay a 'convenience' fee as well as some freedom as you lock yourself on a particular vendor.

## Cloud Concepts Question 5

### The lesson names two situations where the cloud is probably not the right choice. What are they?

If your dataset fits comfortably on a single machine and you do not have massive compute demands.

The learning curve for cloud infrastructure can be very steep. Even doing simple things in the cloud can take a long time, as you have to figure out the right resources and jargon initially.


# Part 2: Warmup — Cloud Landscape

## Cloud Landscape Question 1
### Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.

AWS: the pioneer of the cloud infrastructure, has the broadest service catalog,  in use in large enterprises, start-ups or non-profits with engineering staff.

Microsoft Azure: the dominant provided in enterprise and government settings due to its deep integration with Windows.

Google Cloud Platform (GCP): The strongest in data and machine learning, Google built many of the foundational ideas in modern distributed systems and GCP reflects that heritage, if working on ML infrastructure or Large scale analytics, this is the preferred platform.

## Cloud Landscape Question 2

### The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

The ease of access provided by supabase, as it does not need any organizational provisioning or planning.

Pedagogical fit as supabase stores data as rows and columns in a relational database instead of the opaque files stored by Azure.

Pipeline coherence, as the tool's architecture naturally mirrors the relational data modeling concepts taught in coursework, reinforcing the ETL pipeline workflow.

### Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

When evaluating a cloud tool for a new project, you shouldn't just look at raw enterprise scale or technical capability; you also have to weigh friction, transparency, and alignment with your goals. A tool might be an industry heavyweight, but if its setup overhead slows down your velocity or hides how things work under the hood, a more focused or developer-friendly platform is often the better choice for getting a project off the ground.

## Cloud Landscape Question 3

### For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

You need to store 10 TB of image files and retrieve them by filename from any machine.
- Object Storage, offered by AWS S3.

You need to run an ML training job on a GPU for four hours, then shut it down.
- Virtual Machines, Offered by Azure Virtual Machines.

You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
- Serverless Compute, Offered by AWS Lambda.

You need to send structured data to a large language model and get a text response back.
- LLM API, offered by Open AI API or Google Gemini API.

## Cloud Landscape Question 4

### The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

Project: An app to track gym sessions for powerlifters

Multi-Provider Stack:

Storage & Database (Supabase / BaaS): Stores the historical workouts as well as templates for exercises and workouts, handles authentication.

LLM API (OpenAI API): Generates automated natural-language summaries of weekly progress for users.

Compute & Hosting (Render or Vercel / PaaS): Hosts the interactive Streamlit data visualization web app.

Benefits of Consolidating to One Provider:
Consolidating everything onto a single major cloud provider simplifies billing, streamlines security policies and network routing, and also reduces administrative overhead.

What You Give Up:
You often give up best-of-breed developer velocity and specialized features. For example, while you could host an LLM or a developer-friendly database natively on a hyperscaler, using specialized platforms like OpenAI or Supabase allows you to bypass heavy infrastructure configuration and leverage purpose-built developer experiences that would take significantly more effort to wire up from scratch on a single raw provider.

