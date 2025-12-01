# Poker Advisor API


1) Executive Summary

This project implements a fully containerized poker advisor service that is able to deliver real time preflop recommendations using monte carlo simulations while also being able to tell players their equity's and draws on subsequent streets. the system uses a FASTAPI based HTTP where users are able to input their cards and other optional data such as position, stack, pot size, and action they might be facing. The service is then avle ot compute the expected equity though the repeated random trials, find the opppent distibution, and then return a clear descion for the player based on the different equity thresholds based on the percieved ranges. 



2) System Overview
  A. Course Concepts
    Containerization(Docker) - Deterministic runtime, reproducibility, ability to run locally
    API/Webservices - Creating a structured API design for ML/DS systems
    Simulation Pipeline - Creating a Monte Carlo Statistical model that is able to compute user equity
    Cloud deployment (Azure) - Allows app to be ran virtually
    Data modeling - JSON based opponent hand range specification

  B. The architecture Diagram to this project is found under the assets folder. The poker advisor uses a FastAPI server that is containerized using Docker. The server hosts an endpoint that accepts user poker hand data and optional game context. The server loads predefined opponent hand ranges from JSON files stored in the container. The service uses eval7 to compute the equities. Everything is packaged into a single docker image and deployed to Azure Container Apps for public access, which provides a secure HTTPS endpoint. User interact though a browsser UI, which sends requests to the FastAPI backend.

  C. Data/Models/Services
    Hand Range Data 
      Source: Custom JSON files stores in the files /data/ranges*.json
      Format: JSON
      Size: 5-10 KB each
      License: Self developed **

    Simulation Engine (eval7)
      Source: PyPl (eval package
      Format: Python
      license: MIT license
      Purpose: Fast hand evaluation for Monte Carlo trials 

    FastAPI Application service
      Source: Local code in the /app file
      Framework: FastAPI + Uvicorn
      license: FastAPI (MIT), Uvicorn ( BSD)

    Container Image
      Base: Python 
      Artifacts: FastAPI server, simulation engine, JSON range files   
      Image Size: ~230–260 MB 
      License: Follows base Python + included packages

      Cloud Deployment (Optional)
        Platform: Azure Container Apps
        Public Endpoint: Placeholder (to be replaced)
        Ingress: HTTPS


3) How to Run (Local)

Before Running code make sure youre Docker app is open.

docker build -t poker-advisor:latest .
docker run -p 8000:8000 poker-advisor:latest

You should see 
http://localhost:8000
Once you see this you can copy into browser to use locally.

Health test (Optional)
Visit this page but typing into local browser
http://localhost:8000/docs


4) Design Decisions

Why this concept?

  FastAPI over Flask or Django: I choose FastAPI in this projet becasue it provided automatic request validation with Pydantic and has a high performacne. FLask requies more validaiton and is typically slower undet concurrent workloads. Django was avioded becasue it has a large, monolithic structure that was too complex for the light weight service that was being created.
  Monte Carlo over Game theory solver (GTO): During the creation of this project there was alot of thought about the implemation of game theory and how that would be a possibility. As the project went on I realized that to create and use a true GTO solver it required too much heay GPU hardware and an immense amount of training data. Monte Carlo simulation was then selected as it is able to provide clear simulation based approache that only relied on compuational worloads that run off CPU, requires no pretraining, and still produces reliable preflop equity estimates.
  Why JSON files: JSON files where used becasue it allows the system to stay simple and easy to change. User can direcuty edi the JSON to costimize ranges without touching code. Using a database was not chosen because it would create alot of overhead and networking complexity.

Tradeoffs
  Monte Carlo vs GTO solver: In the devoplment I wanted to create a true game theory service but that proved unreliable with too much to run on a simple, clean service which is why there was a trade off using Monte Carlo Simulations instead. Monte Carlo are fast and are able to be ran on CPU wihtout any GPU hardware.
  Preflop only scope: I had to sacrific full hand modeling and complex multi street advice in order to create a more reliable preflop engine that is easy to build and deploy in a fast timeline. The service now can tell you youre equity and outs but is unable to provide advice to what decsion to make post flop.
  JSON files vs Database: With the use of JSON there is no ability to have version hsitory or runtime editing however it allowed for zero operational overhead, easier deployment, and very easy customization of ranges.
  CPU only vs GPU: With the service currently provided by the site it only has to run off CPU which is cost friendly, portable, and can be used bu any laptop. GPU allows us to account for more simulations and perhaps a more accurate bot but would not be able to be used by many users.

Secuirty and Privacy

This service does not handle any user personal infomation, however, there are still ways to ernsure safe operation. There has to be a input validation which is means that the API and Pydantic must validate all entries becasue running, rejecting all improper submissions. The site always keeps the privacy of the user by holding no user PII. The system also aviouds exposing internal stack traces or sesnetive reuntime details while also having an isolated execuation with docker isolating the envournemtn and preventing depednedancy conflcts.

Ops (Logging, Metrics, Scaling & Limitations)

Logging

FastAPI middleware can be used to track request metadata, simulation runtime, and error events. Logs exclude any sensitive content and focus on operational health.

Metrics

Azure Container Apps provides built-in metrics for CPU usage, memory consumption, request count, and container restarts. Future work could add custom simulation-level metrics.

Scaling

The stateless architecture enables horizontal scaling easily. Azure Container Apps can automatically scale replicas up or down based on CPU usage or request load.

Operational Limitations

  Simulation is CPU-bound; large simulation counts increase response times.

  No caching means identical requests recompute simulations, raising compute cost in high-traffic settings.

  Single-container deployment means compute and API share CPU resources.

  No rate-limiting by default (could be added via a proxy or FastAPI extension).

  JSON range files are loaded at runtime; malformed JSON causes startup failure (could be mitigated with schema validation).


5) Results and Evaluation

There is a screenshot of the output in the assets folder.

Correctness Validation
Known equities: Verified results for common hands (e.g., AhKh vs JJ, AKs vs AQo) match published preflop equity tables within expected Monte Carlo variance (±0.5–1.0%).

Range sampling: Ensured JSON range distributions result in correctly weighted opponent hand samples.
Repeatability: For fixed RNG seeds, results remain consistent across runs.
API Validation
Input validation:

  Invalid card formatting returns 400 Bad Request. 
  Unknown range names produce safe error messages (no stack traces).
  Health check: /health endpoint reliably returns service readiness.
  Load testing: Service remained stable under repeated bursts of 20–50 sequential requests.

Tests: 
 A. The first test in the project is smoke_check.py which can be ran by running cd tests then python3.11 smoke_check.py in the terminal which should output AA equity with a green check mark signifiying that it passed the smoke test and it is running correctly.

 B. The second test in the project is the deterministic test which can be rean by pasting cd test then  python3.11 -m pytest test_deterministic.py -q in the terminal

C. Test_ranges.py. cd tests then python3.11 -m pytest test_ranges.py -q which should print out 1 passed

6. What’s Next

    Going forwards, the main aspect that could be improved is the implemenation of postflop adivce as well as inclduing more complex modeling such as GTO solver which would be the end goal for the service. GTO is the key to poker strategy and would allow for a more robust and complete poker advisor. Being a GTO solver would create the ability to provide advice on all streets and be able to adjust to different opponent types. A strech feature I would like the ability to add is multi oppenent modeling so that the user can input more than one opponent and get a more accurate equity estimate based on multiple players at the table.

7)

GitHub Repo:

Public Cloud App:


Cloud Deployment (Extra Credit)

Log Into Azure 
  az login

Create Resource Group
  az group create --name poker-test-rg --location northcentralus

Create Azure Container Registry (ACR)

  az acr create \
    --resource-group poker-test-rg \
    --name pokerregistrytestwh \
    --sku Basic \
    --location northcentralus
    
Log in to the new registry

  az acr login --name pokerregistrytestwh

Build + Tag + Push Docker image

  docker build -t poker-advisor-test .

  docker tag poker-advisor-test pokerregistrytestwh.azurecr.io/poker-advisor:v2

  docker push pokerregistrytestwh.azurecr.io/poker-advisor:v2

Ensure required Azure providers

  az provider register --namespace Microsoft.App --wait
  az provider register --namespace Microsoft.OperationalInsights --wait

Create a NEW Container App environment

  az containerapp env create \
    --name poker-test-env \
    --resource-group poker-test-rg \
    --location northcentralus

Deploy the app
  az containerapp create \
    --name poker-test-app \
    --resource-group poker-test-rg \
    --environment poker-test-env \
    --image pokerregistrytestwh.azurecr.io/poker-advisor:v2 \
    --target-port 8000 \
    --ingress external \
    --registry-server pokerregistrytestwh.azurecr.io \
    --registry-identity system

Get the Pulbic URL
    az containerapp show \
      --name poker-test-app \
      --resource-group poker-test-rg \
      --query properties.configuration.ingress.fqdn \
      --output tsv


This project also implements a Observability/CI build with Uvicorn logging and Pytest based testing suite to ensure the code is running correctly. There is also Docker logs  which show youre request traces and startup events. When the app is running on Uvicorn you can see all the request logs in the terminal where you started the docker container.


  

