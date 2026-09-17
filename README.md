# TET4565 — Power Market Optimisation

Welcome to the course repository for **TET4565**. The repository supports the course with Python/Pyomo examples, project guidance, tutorials, lecture-related resources, and a practical GitHub/VS Code workflow.

## Repository structure

```text
TET4565-course-github/
├── code-examples/
│   ├── L02-Energy-Reserve-Optimisation/
│   ├── L02-Stochastic-Market-Clearing/
│   ├── L03-Benders-Example/
│   ├── L03-Benders-Example-Compact/
│   ├── L03-Stochastic-Benders-Decomposition/
│   └── Battery-Optimisation/
├── lectures/
├── tutorials/
├── project/
│   ├── part-1/
│   └── templates/
├── resources/
└── .github/
```

## Python code examples

The repository currently includes examples on:

- energy and reserve-capacity co-optimisation;
- two-stage stochastic market clearing;
- stochastic market clearing with an explicit reserve requirement;
- introductory and compact Benders decomposition;
- stochastic Benders decomposition, including a larger ten-generator case;
- battery optimisation across day-ahead and reserve/balancing markets with CVaR risk modelling and an efficient-frontier analysis.

See [`code-examples/README.md`](code-examples/README.md) for the code index and instructions.

## Project work

The course project focuses on **power market optimisation under uncertainty**, including two-stage stochastic optimisation, model implementation and comparison, and decomposition methods.

Students should review the project material released in Canvas under:

**PMO Interim Submission – Project Part 1**

Please also register for a project group in Canvas under:

**People → Groups → TET4565 – Course Project Groups**

Project groups are limited to **3 students**.

## Project Consultancy Sessions

Please use the Project Consultancy Sessions actively for guidance, questions, troubleshooting, and discussion of modelling choices.

The recording of the GitHub / VS Code consultancy session is available here:

https://ntnu.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=ff7f3f31-46d2-4a93-aa15-b4bf00c867b7#

## Getting started

Create the course environment:

```bash
conda env create -f environment.yml
conda activate tet4565
```

or install the Python packages with:

```bash
pip install -r requirements.txt
```

The optimisation examples currently use **Gurobi**, so a working Gurobi installation and licence are required to solve them.

## Recommended Git workflow

For normal group work:

```text
PULL → EDIT → ADD → COMMIT → PUSH
```

For larger changes:

```text
BRANCH → COMMIT → PUSH → PULL REQUEST → REVIEW → MERGE
```

Good practice is to pull before starting, make small focused commits, use clear commit messages, and avoid having several people edit the same file at the same time when possible.

## Questions and support

Use the Canvas discussion board for questions that may benefit the whole class. For group-specific or personal questions, use your group discussion area or contact the teaching team.

The earlier you start working with the project material, the more effectively we can support you during the consultancy sessions.
