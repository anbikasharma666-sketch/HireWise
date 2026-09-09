"""
generate_data.py
Generates realistic, professional mock PDF documents for HireWise:
1. data/handbook.pdf (HireWise Technologies Employee Handbook & Benefits)
2. data/job_description.pdf (Software Engineer Job Description)
3. data/resume_candidate_1.pdf (Candidate 1 Resume)
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def build_pdf(filename, title, sections):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        fontName='Helvetica-Bold',
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e3a8a'),
        fontName='Helvetica-Bold',
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph("HireWise Technologies Inc. - Internal Human Resources Document", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceAfter=12))

    for sec_title, sec_content in sections:
        if sec_title:
            story.append(Paragraph(sec_title, h2_style))
        if isinstance(sec_content, list):
            for item in sec_content:
                if item.startswith("-") or item.startswith("*"):
                    story.append(Paragraph(f"&bull; {item.lstrip('-* ')}", bullet_style))
                else:
                    story.append(Paragraph(item, body_style))
        else:
            story.append(Paragraph(sec_content, body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"[OK] Successfully generated {filename}")

def generate_handbook():
    sections = [
        ("1. Company Mission and Work Culture", [
            "Welcome to HireWise Technologies! We build intelligent talent acquisition and human resource automation systems.",
            "Our core philosophy emphasizes transparency, continuous learning, engineering excellence, and mutual respect."
        ]),
        ("2. Working Hours and Core Collaboration Hours", [
            "Standard working hours are Monday through Friday, 9:30 AM to 6:30 PM (local time).",
            "Core collaboration hours are 10:30 AM to 4:30 PM, during which all team members are expected to be available for synchronous meetings, standups, and communications.",
            "Employees are allowed flexible starting times between 8:30 AM and 10:00 AM with prior notification to their engineering manager."
        ]),
        ("3. Work-From-Home (WFH) and Hybrid Policy", [
            "HireWise operates under a structured hybrid working model.",
            "Eligible full-time employees can work remotely up to 2 days per week with manager approval.",
            "Mondays and Thursdays are designated as team in-office collaboration days.",
            "Employees must have a stable internet connection and quiet workspace during remote days."
        ]),
        ("4. Paid Leave and Time-Off Policy", [
            "Employees are granted 18 days of paid annual leave per calendar year, accrued monthly at 1.5 days per month.",
            "Additionally, employees receive 10 days of paid sick and casual leave per year.",
            "Unused paid annual leave up to a maximum of 8 days can be carried forward into the next calendar year.",
            "Leave requests exceeding 3 consecutive days must be submitted at least 2 weeks in advance."
        ]),
        ("5. Public Holidays", [
            "HireWise observes 10 paid official public holidays each year, published annually by the HR operations department in December."
        ]),
        ("6. Health Insurance and Medical Benefits", [
            "HireWise provides comprehensive medical insurance coverage up to $500,000 (5 Lakhs INR equivalent) per employee and immediate dependents (spouse and children).",
            "The plan includes outpatient consultations, hospitalization, prescription drug coverage, and annual preventive health checkups.",
            "Dental and vision insurance riders are fully sponsored by the company up to $500 per year."
        ]),
        ("7. Additional Employee Benefits and Perks", [
            "- Annual Learning Stipend: $1,000 per employee annually for professional courses, conferences, and certifications.",
            "- Health & Wellness Allowance: $50 per month for gym memberships, fitness trackers, or mental wellness subscriptions.",
            "- Equipment Allowance: Brand new developer laptop (MacBook Pro or ThinkPad P-Series) and home monitor setup.",
            "- Daily Catered Meals and Healthy Snacks provided at all office locations."
        ]),
        ("8. Probation Period and Confirmation", [
            "The standard probation period for newly joined software engineers and full-time employees is 6 months from their joining date.",
            "A formal performance review will be conducted at month 5 by the engineering manager and HR partner.",
            "Upon successful completion, employees receive an official written confirmation letter."
        ]),
        ("9. Notice Period Policy", [
            "During the probation period, the notice period required for separation is 15 calendar days from either party.",
            "Following official confirmation, the standard notice period is 30 calendar days for individual contributors.",
            "Notice periods may be shortened or waived only with mutual consent and executive HR sign-off."
        ]),
        ("10. Basic Onboarding Process", [
            "- Day 1: IT workstation setup, HR orientation, email and SSO access provisioning.",
            "- Week 1: Peer buddy assignment, architecture walkthroughs, dev environment setup, and first commit.",
            "- 30-Day Checkpoint: First sprint retrospective and initial 1-on-1 manager alignment.",
            "- 60-Day and 90-Day Milestones: In-depth project contribution review and skill development roadmap."
        ])
    ]
    build_pdf("data/handbook.pdf", "HireWise Technologies - Employee Handbook", sections)

def generate_job_description():
    sections = [
        ("Position Overview", [
            "Job Title: Software Engineer",
            "Department: Core Engineering & Platform Infrastructure",
            "Location: Hybrid (2 days WFH / 3 days In-Office)",
            "Employment Type: Full-Time"
        ]),
        ("About the Role", [
            "We are looking for an ambitious and talented Software Engineer to design, develop, and scale high-performance microservices and backend APIs that power our next-generation hiring platform."
        ]),
        ("Required Skills & Qualifications", [
            "- Python: Strong proficiency in Python 3.x, OOP, clean architecture, and modern libraries.",
            "- Java: Solid understanding of core Java, multithreading, and enterprise backend fundamentals.",
            "- SQL & Databases: Proficiency in relational database design, query optimization, indexing, and PostgreSQL/MySQL.",
            "- Git: Experience with Git version control, branching strategies, PR workflows, and code hygiene.",
            "- REST APIs: Demonstrated experience designing, building, and documenting resilient RESTful APIs.",
            "- Data Structures & Algorithms: Strong foundations in complexity analysis (Big-O), search, sorting, and graph/tree structures."
        ]),
        ("Preferred Skills", [
            "- Docker: Containerization, Dockerfile optimization, and multi-container setups.",
            "- AWS: Cloud services experience with AWS ECS, S3, EC2, Lambda, or RDS.",
            "- React: Familiarity with modern frontend development, TypeScript, and React hooks.",
            "- CI/CD: Automated build and deployment pipelines using GitHub Actions or GitLab CI.",
            "- Unit Testing: Strong commitment to test-driven development, pytest, JUnit, and mock frameworks."
        ]),
        ("Experience Requirements", [
            "- 1+ year of software development experience (including substantial software engineering internships or significant production-grade open-source/academic projects)."
        ]),
        ("Key Responsibilities", [
            "- Design and develop robust, modular backend services and RESTful APIs.",
            "- Optimize database schemas, queries, and caching mechanisms for scalability.",
            "- Collaborate with cross-functional teams, participate in peer code reviews, and maintain clean Git branches.",
            "- Debug complex production issues, profile latency bottlenecks, and ensure 99.9% uptime.",
            "- Write comprehensive technical documentation and unit test suites."
        ]),
        ("Education Requirements", [
            "- Bachelor degree in Computer Science, Information Technology, or equivalent practical experience."
        ])
    ]
    build_pdf("data/job_description.pdf", "Job Description - Software Engineer", sections)

def generate_resume():
    sections = [
        ("Candidate Profile", [
            "Candidate: Candidate 1",
            "Target Role: Software Engineer",
            "Email: candidate1.hirewise.demo@example.com | Phone: (555) 019-2834 | Location: Bangalore / Remote"
        ]),
        ("Education", [
            "- Bachelor of Technology in Computer Science & Engineering",
            "  Graduation Year: 2025 | CGPA: 8.6 / 10.0",
            "  Relevant Coursework: Data Structures & Algorithms, Database Management Systems, Object-Oriented Programming, Computer Networks, Operating Systems"
        ]),
        ("Technical Skills", [
            "- Programming Languages: Python, Java, SQL, C++",
            "- Frameworks & Web: REST APIs, FastAPI, Flask, Basic React, HTML5, CSS3",
            "- Databases: PostgreSQL, MySQL, SQLite",
            "- Developer Tools: Git, GitHub, Linux Shell, VS Code, Postman",
            "- Core Concepts: Object-Oriented Design, Data Structures & Algorithms, Schema Design"
        ]),
        ("Professional Experience", [
            "Software Engineering Intern | TechNova Solutions (6-Month Internship)",
            "- Developed scalable RESTful backend services using Python and FastAPI, handling 15,000+ daily requests.",
            "- Designed and optimized PostgreSQL relational database schemas, reducing query latency by 28%.",
            "- Built Java-based automated batch data validation utilities, ensuring 99.5% data consistency.",
            "- Collaborated with senior engineers using Git version control and actively participated in weekly code reviews."
        ]),
        ("Key Projects", [
            "1. Inventory Management REST API (Python, SQL, REST APIs, Git)",
            "- Engineered a modular inventory tracking backend with JWT authentication and CRUD endpoints.",
            "- Implemented relational schema with foreign key constraints, indexes, and automated migration scripts.",
            "- Authored comprehensive Postman API documentation and achieved 92% unit test coverage using pytest.",
            "",
            "2. Student Academic Portal (Java, SQL, Basic React, Git)",
            "- Created a full-stack student management application featuring course registration and grading modules.",
            "- Implemented core business logic in Java with JDBC integration to MySQL database.",
            "- Built responsive dashboard UI using React components and integrated REST endpoints."
        ]),
        ("Certifications & Achievements", [
            "- Top 10% in University Algorithmic Coding Challenge (Data Structures & Algorithms).",
            "- Certified Python Associate Developer."
        ])
    ]
    build_pdf("data/resume_candidate_1.pdf", "Candidate Resume - Candidate 1", sections)

if __name__ == "__main__":
    ensure_data_dir()
    generate_handbook()
    generate_job_description()
    generate_resume()
    print("[OK] All mock PDF documents generated successfully in data/ directory.")
