## Objectives and How They Are Addressed

### 1. Data Simulation

A synthetic **weekly marketing and sales dataset** is generated, including:

- Channel spend (e.g., TikTok, Facebook, Google Ads)
- Observed sales outcomes

The simulated data incorporates realistic marketing dynamics:

- Intermittent spend patterns  
- Lagged media effects (carryover / adstock)  
- Diminishing returns (saturation)  
- Seasonality and baseline trends  

Using synthetic data enables:

- Controlled validation of model behavior  
- Clear interpretation of channel-level effects  
- Full reproducibility of results  

---

### 2. Model Construction (Bayesian MMM)

The core MMM is implemented using **PyMC** and reflects common industry best practices.

#### Model components include:

**Baseline terms**
- Intercept  
- Linear trend  
- Seasonal effects (Fourier series)  

**Media transformations**
- Geometric adstock to capture carryover effects  
- Hill-type saturation functions to model diminishing returns  

**Bayesian formulation**
- Informative but flexible priors for all parameters  
- Posterior inference via MCMC sampling  
- Full uncertainty quantification for parameters and channel contributions  

This structure balances interpretability, flexibility, and alignment with production-grade MMM approaches.

---

### 3. Model Insights

From the fitted model, the following insights are derived:

- Channel-level incremental sales contributions  
- **Return on Ad Spend (ROAS)** by channel  
- Relative channel effectiveness  
- Budget allocation implications, including:
  - Channels with high ROI but limited scale  
  - Potentially oversaturated channels  
  - Opportunities for marginal budget reallocation  

Model outputs are saved as CSV files and surfaced through a lightweight **Flask** application for monitoring model performance (internal use) and easy inspection (stakeholders).

---

### 4. Containerization

The full workflow is containerized using **Docker**, enabling reproducible execution across environments.

Key aspects include:

- Python environment and dependencies defined in `requirements.txt`  
- Application code and model artifacts packaged into a single image  
- The app exposing results 

---

## Build and Push the Docker Image

To build and publish the Docker image locally:

```bash
# Build the image
docker build -t mmm-app2 .

# Authenticate with Docker Hub
docker login

# Tag the image
docker tag mmm-app2 qzhou333/mmm-app2:latest

# Push to Docker Hub
docker push qzhou333/mmm-app2:latest

# References

- Solomons, D., Kliphuis, T., & Wadley, M. (2022).  
  *eMarketing: The Essential Guide to Marketing in a Digital World* (7th ed.).  
  Red & Yellow Creative School of Business.

- OpenAI. (2026).  
  *ChatGPT (5.2)* [Large language model].

