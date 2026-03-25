# MLOps

**MLOps** or **ML Ops** is a paradigm that aims to deploy and maintain machine learning models in production reliably and efficiently. It bridges the gap between machine learning development and production operations, ensuring that models are robust, scalable, and aligned with business goals. The word is a compound of "machine learning" and the continuous delivery practice (CI/CD) of DevOps in the software field. Machine learning models are tested and developed in isolated experimental systems. When an algorithm is ready to be launched, MLOps is practiced between Data Scientists, DevOps, and Machine Learning engineers to transition the algorithm to production systems. Similar to DevOps or DataOps approaches, MLOps seeks to increase automation and improve the quality of production models, while also focusing on business and regulatory requirements. While MLOps started as a set of best practices, it is slowly evolving into an independent approach to ML lifecycle management. MLOps applies to the entire lifecycle - from integrating with model generation (software development lifecycle, continuous integration/continuous delivery), orchestration, and deployment, to health, diagnostics, governance, and business metrics. 

## Definition

MLOps is a paradigm, including aspects like best practices, sets of concepts, as well as a development culture when it comes to the end-to-end conceptualization, implementation, monitoring, deployment, and scalability of machine learning products. Most of all, it is an engineering practice that leverages three contributing disciplines: machine learning, software engineering (especially DevOps), and data engineering. MLOps is aimed at productionizing machine learning systems by bridging the gap between development (Dev) and operations (Ops). Essentially, MLOps aims to facilitate the creation of machine learning products by leveraging these principles: CI/CD automation, workflow orchestration, reproducibility; versioning of data, model, and code; collaboration; continuous ML training and evaluation; ML metadata tracking and logging; continuous monitoring; and feedback loops. 

## History

Interest in operationalizing machine learning systems began to grow in the mid-2010s as ML projects started moving from experimentation to production use. The challenges associated with sustaining such systems were highlighted in a 2015 paper. The predicted growth in machine learning included an estimated doubling of ML pilots and implementations from 2017 to 2018, and again from 2018 to 2020. MLOps rapidly began to gain traction among AI/ML experts, companies, and technology journalists as a solution that can address the complexity and growth of machine learning in businesses. 

Reports show a majority (up to 88%) of corporate machine learning initiatives are struggling to move beyond test stages. However, those organizations that actually put machine learning into production saw a 3–15% profit margin increases. The MLOps market size was USD 2,191.8 Million in 2024, and is projected to be USD 16,613.4 Million in 2030. 

## Architecture

Machine Learning systems can be categorized in eight different categories: data collection, data processing, feature engineering, data labeling, model design, model training and optimization, endpoint deployment, and endpoint monitoring. Each step in the machine learning lifecycle is built in its own system, but requires interconnection. These are the minimum systems that enterprises need to scale machine learning within their organization. 

## Goals

There are a number of goals enterprises want to achieve through MLOps systems successfully implementing ML across the enterprise, including: 

 * Deployment and automation
 * Reproducibility of models and predictions
 * Diagnostics
 * Governance and regulatory compliance
 * Scalability
 * Collaboration
 * Business uses
 * Monitoring and management

A standard practice, such as MLOps, takes into account each of the aforementioned areas, which can help enterprises optimize workflows and avoid issues during implementation. 

Vendors such as Adaptive ML deliver commercial reinforcement learning operations (RLOps) and MLOps-infrastructure, targeting organizations deploying large language models in production. 

A common architecture of an MLOps system would include data science platforms where models are constructed and the analytical engines where computations are performed, with the MLOps tool orchestrating the movement of machine learning models, data and outcomes between the systems. 

 * ModelOps, according to Gartner, MLOps is a subset of ModelOps. MLOps is focused on the operationalization of ML models, while ModelOps covers the operationalization of all types of AI models.
 * AIOps, a similarly named, but different concept - using AI (ML) in IT and Operations.

## References

 1. Talagala, Nisha. "Why MLOps (and not just ML) is your Business' New Competitive Frontier". _AITrends_. Archived from the original on 19 January 2021. Retrieved 30 January 2018.
 2. Kreuzberger, Dominik; Kühl, Niklas; Hirschl, Sebastian (2023). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture". _IEEE Access_. **11** : 31866–31879\. arXiv:2205.02302. Bibcode:2023IEEEA..1131866K. doi:10.1109/ACCESS.2023.3262138. ISSN 2169-3536. S2CID 248524628.
 3. Sculley, D.; Holt, Gary; Golovin, Daniel; Davydov, Eugene; Phillips, Todd; Ebner, Dietmar; Chaudhary, Vinay; Young, Michael; Crespo, Jean-Francois; Dennison, Dan (7 December 2015). "Hidden Technical Debt in Machine Learning Systems" (PDF). _NIPS Proceedings_ (2015). Retrieved 14 November 2017.
 4. Sallomi, Paul; Lee, Paul. "Deloitte Technology, Media and Telecommunications Predictions 2018" (PDF). _Deloitte_. Retrieved 13 October 2017.
 5. https://www.meetup.com/MLOps-Silicon-Valley/?_cookie-check=o1SkbKRfUlSuQoT3
 6. Bughin, Jacques; Hazan, Eric; Ramaswamy, Sree; Chui, Michael; Allas, Tera; Dahlström, Peter; Henke, Nicolaus; Trench, Monica. "Artificial Intelligence The Next Digital Frontier?". _McKinsey_. McKinsey Global Institute. Retrieved 1 June 2017.
 7. Grand View Research. "MLOps Market Size, Share & Trends Analysis Report By Component (Platform, Service), By Deployment (Cloud, On-premises), By Organization Size, By Vertical (BFSI, Retail & E-commerce), By Region, And Segment Forecasts, 2025 - 2030". _grandviewresearch.com_. Retrieved 3 July 2025.
 8. Walsh, Nick. "The Rise of Quant-Oriented Devs & The Need for Standardized MLOps". _Slides_. Nick Walsh. Retrieved 1 January 2018.
 9. "Code to production-ready machine learning in 4 steps". _DAGsHub Blog_. 2021-02-03. Retrieved 2021-02-19.
 10. Warden, Pete. "The Machine Learning Reproducibility Crisis". _Pete Warden's Blog_. Pete Warden. Retrieved 19 March 2018.
 11. Vaughan, Jack. "Machine learning algorithms meet data governance". _SearchDataManagement_. TechTarget. Retrieved 1 September 2017.
 12. Lorica, Ben. "How to train and deploy deep learning at scale". _O'Reilly_. Retrieved 15 March 2018.
 13. Garda, Natalie. "IoT and Machine Learning: Why Collaboration is Key". _IoT Tech Expo_. Encore Media Group. Retrieved 12 October 2017.
 14. Manyika, James. "What's now and next in analytics, AI, and automation". _McKinsey_. McKinsey Global Institute. Retrieved 1 May 2017.
 15. Haviv, Yaron. "MLOps Challenges, Solutions and Future Trends". _Iguazio_. Retrieved 19 February 2020.
