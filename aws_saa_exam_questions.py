"""
AWS Certified Solutions Architect – Associate (SAA-C03) simulated exam question bank.

65 scenario-based questions (medium to hard), weighted to the official domain split:
    Domain 1 – Design Secure Architectures            30%  -> 20 questions
    Domain 2 – Design Resilient Architectures         26%  -> 17 questions
    Domain 3 – Design High-Performing Architectures   24%  -> 15 questions
    Domain 4 – Design Cost-Optimized Architectures    20%  -> 13 questions

Each question:
    id          unique int
    domain      one of DOMAINS
    difficulty  "Medium" | "Hard"
    question    scenario text
    options     list of answer choices
    answer      list of correct option indexes (len > 1 => multiple response)
    explanation why the answer is correct / why the distractors are wrong
"""

SECURE = "Design Secure Architectures"
RESILIENT = "Design Resilient Architectures"
PERFORMANCE = "Design High-Performing Architectures"
COST = "Design Cost-Optimized Architectures"

DOMAINS = [SECURE, RESILIENT, PERFORMANCE, COST]

QUESTIONS = [
    # ─────────────────────────────────────────────────────────────
    # DOMAIN 1 – DESIGN SECURE ARCHITECTURES (20)
    # ─────────────────────────────────────────────────────────────
    {
        "id": 1, "domain": SECURE, "difficulty": "Medium",
        "question": "A company has separate AWS accounts for development and production. Developers in the development account need read-only access to a specific S3 bucket in the production account. The security team requires least privilege and does not want long-term credentials created in the production account. What should a solutions architect do?",
        "options": [
            "Create IAM users in the production account for each developer and grant them read access to the bucket.",
            "Create an IAM role in the production account with read access to the bucket and a trust policy that allows the development account to assume it. Grant developers sts:AssumeRole permission for that role.",
            "Generate access keys for an IAM user in the production account and store them in AWS Secrets Manager in the development account.",
            "Add a bucket policy that allows s3:GetObject for Principal \"*\" with a condition on the developers' source IP addresses.",
        ],
        "answer": [1],
        "explanation": "Cross-account IAM roles provide temporary credentials via STS and keep access scoped to exactly what the role allows. IAM users and shared access keys create long-term credentials, and a wildcard principal is overly permissive.",
    },
    {
        "id": 2, "domain": SECURE, "difficulty": "Medium",
        "question": "A healthcare company stores patient documents in Amazon S3. Compliance requires encryption at rest, an audit trail of every use of the encryption key, and the ability for the security team to disable the key independently of the S3 administrators. Which solution meets these requirements?",
        "options": [
            "Server-side encryption with Amazon S3 managed keys (SSE-S3).",
            "Server-side encryption with AWS KMS customer managed keys (SSE-KMS) with a key policy controlled by the security team.",
            "Server-side encryption with customer-provided keys (SSE-C).",
            "Client-side encryption with keys stored as objects in a separate S3 bucket.",
        ],
        "answer": [1],
        "explanation": "SSE-KMS with a customer managed key logs every key use in CloudTrail and the key policy enables separation of duties (the security team can disable/rotate the key). SSE-S3 gives no key-level control or audit; SSE-C means AWS doesn't store or audit the key.",
    },
    {
        "id": 3, "domain": SECURE, "difficulty": "Medium",
        "question": "EC2 instances in private subnets must upload data to Amazon S3 in the same Region. The security team requires that this traffic never traverses the internet and that access is restricted to a single bucket. Which solution meets these requirements with the LEAST operational overhead?",
        "options": [
            "Deploy a NAT gateway in a public subnet and restrict the instances' security group outbound rules to S3 IP ranges.",
            "Create an S3 gateway VPC endpoint, add it to the private subnets' route tables, and attach an endpoint policy that allows access only to the required bucket.",
            "Create an AWS Site-to-Site VPN connection and route S3 traffic through it.",
            "Attach an internet gateway and use S3 Transfer Acceleration with a bucket policy.",
        ],
        "answer": [1],
        "explanation": "A gateway endpoint keeps S3 traffic on the AWS network, requires no extra infrastructure, and endpoint policies can restrict which buckets are reachable. NAT gateways and IGWs route through the public internet path.",
    },
    {
        "id": 4, "domain": SECURE, "difficulty": "Medium",
        "question": "An application running on Amazon ECS connects to an Amazon Aurora PostgreSQL database. The security team requires that database credentials are never stored in code or container images and are rotated automatically every 30 days. What should a solutions architect recommend?",
        "options": [
            "Store the credentials in AWS Systems Manager Parameter Store as a String parameter and rotate them with a cron job on an EC2 instance.",
            "Store the credentials in AWS Secrets Manager with automatic rotation enabled and have the ECS task retrieve the secret at runtime using its task IAM role.",
            "Encrypt the credentials with AWS KMS and pass the ciphertext as an environment variable in the task definition.",
            "Store the credentials in an encrypted S3 object and download it during container startup.",
        ],
        "answer": [1],
        "explanation": "Secrets Manager provides native, managed rotation for Aurora/RDS credentials and integrates with ECS task roles. The other options require custom rotation logic or still embed secrets in configuration.",
    },
    {
        "id": 5, "domain": SECURE, "difficulty": "Medium",
        "question": "A public web application behind an Application Load Balancer is being targeted by SQL injection and cross-site scripting (XSS) attacks. Which AWS service should be used to block these requests before they reach the application?",
        "options": [
            "AWS Shield Standard",
            "Amazon GuardDuty",
            "AWS WAF with managed rule groups associated with the ALB",
            "Network ACLs on the ALB subnets",
        ],
        "answer": [2],
        "explanation": "AWS WAF inspects Layer 7 HTTP requests and its managed rule groups block common exploits such as SQLi and XSS. Shield protects against DDoS, GuardDuty detects threats but does not block requests, and NACLs work at Layer 3/4.",
    },
    {
        "id": 6, "domain": SECURE, "difficulty": "Hard",
        "question": "A company uses AWS Organizations with more than 100 member accounts. Due to data residency regulations, no resources may be created outside the eu-west-1 and eu-central-1 Regions in any account, including by account administrators. Global services such as IAM must continue to work. What is the MOST effective solution?",
        "options": [
            "Create an IAM policy in each account that denies actions outside the approved Regions and attach it to all users and roles.",
            "Apply a service control policy (SCP) at the root OU that denies all actions when aws:RequestedRegion is not one of the approved Regions, excluding global services.",
            "Enable AWS Config in every account with a rule that flags resources created in unapproved Regions.",
            "Use AWS CloudTrail and an EventBridge rule to invoke a Lambda function that deletes resources created in unapproved Regions.",
        ],
        "answer": [1],
        "explanation": "SCPs set the maximum permissions for every principal in member accounts, including account administrators, and are centrally managed. Config and CloudTrail approaches are detective/reactive, and per-account IAM policies can be removed by local admins.",
    },
    {
        "id": 7, "domain": SECURE, "difficulty": "Medium",
        "question": "A company serves static website content from an S3 bucket through Amazon CloudFront. Users can currently bypass CloudFront by accessing the S3 object URLs directly. How should a solutions architect ensure content is accessible ONLY through CloudFront?",
        "options": [
            "Enable S3 static website hosting and restrict access with a Referer header condition.",
            "Configure CloudFront signed URLs for all objects.",
            "Configure an origin access control (OAC) on the CloudFront distribution and update the bucket policy to allow access only from that distribution; block all public access on the bucket.",
            "Enable S3 Transfer Acceleration and disable the standard S3 endpoint.",
        ],
        "answer": [2],
        "explanation": "OAC lets CloudFront sign requests to S3 and the bucket policy grants access only to the CloudFront service principal for that distribution. Signed URLs control which viewers can access content, not whether S3 can be reached directly.",
    },
    {
        "id": 8, "domain": SECURE, "difficulty": "Medium",
        "question": "A security audit found that some applications upload objects to an S3 bucket using HTTP. The company must ensure that ALL requests to the bucket are encrypted in transit. What should a solutions architect do?",
        "options": [
            "Enable default encryption with SSE-S3 on the bucket.",
            "Add a bucket policy statement that denies all s3:* actions when the condition aws:SecureTransport is false.",
            "Enable S3 Object Lock on the bucket.",
            "Configure a VPC endpoint policy that requires encryption.",
        ],
        "answer": [1],
        "explanation": "The aws:SecureTransport condition key evaluates to false for non-TLS requests, so a Deny statement rejects all HTTP traffic. Default encryption covers data at rest, not in transit.",
    },
    {
        "id": 9, "domain": SECURE, "difficulty": "Hard",
        "question": "A mobile app needs users to sign up and sign in with email or social identity providers. After signing in, users must upload photos directly to their own prefix in an S3 bucket using temporary AWS credentials. Which combination of services meets these requirements?",
        "options": [
            "Amazon Cognito user pools for authentication and Amazon Cognito identity pools to exchange tokens for temporary IAM credentials scoped with policy variables.",
            "IAM users for each mobile user with access keys embedded in the app.",
            "AWS IAM Identity Center with permission sets for each mobile user.",
            "Amazon API Gateway with API keys that map to S3 bucket prefixes.",
        ],
        "answer": [0],
        "explanation": "User pools handle sign-up/sign-in and federation with social IdPs; identity pools vend temporary STS credentials, and IAM policy variables (cognito-identity.amazonaws.com:sub) can restrict each user to their own prefix. IAM Identity Center is for workforce users, not app customers.",
    },
    {
        "id": 10, "domain": SECURE, "difficulty": "Medium",
        "question": "A company wants continuous, intelligent threat detection that alerts when EC2 instances communicate with known command-and-control IP addresses or when there is unusual API activity, using VPC Flow Logs, DNS logs, and CloudTrail events, without deploying agents. Which service should be used?",
        "options": ["Amazon Inspector", "Amazon GuardDuty", "Amazon Macie", "AWS Config"],
        "answer": [1],
        "explanation": "GuardDuty analyzes VPC Flow Logs, DNS logs and CloudTrail events using threat intelligence and ML. Inspector scans for software vulnerabilities, Macie discovers sensitive data in S3, and Config tracks configuration compliance.",
    },
    {
        "id": 11, "domain": SECURE, "difficulty": "Medium",
        "question": "A company has thousands of S3 buckets across many accounts and must identify which buckets contain personally identifiable information (PII) such as passport numbers and credit card numbers. Which solution requires the LEAST effort?",
        "options": [
            "Use Amazon Athena to run SQL queries with regular expressions against every bucket.",
            "Enable Amazon Macie with a delegated administrator account and run automated sensitive data discovery.",
            "Use Amazon Inspector to scan all S3 buckets.",
            "Write a Lambda function triggered by S3 events that scans each new object with regular expressions.",
        ],
        "answer": [1],
        "explanation": "Macie uses managed data identifiers and ML to discover PII in S3 and works across an organization through a delegated administrator. The other options require custom code or do not inspect S3 object content.",
    },
    {
        "id": 12, "domain": SECURE, "difficulty": "Medium",
        "question": "An application running on EC2 instances must read and write items in an Amazon DynamoDB table. What is the MOST secure way to grant the application access?",
        "options": [
            "Store an IAM user's access keys in the application configuration file on the instance.",
            "Create an IAM role with a least-privilege policy for the table and attach it to the instances through an instance profile.",
            "Store an IAM user's access keys in the instance user data.",
            "Use the root user's access keys stored in AWS Secrets Manager.",
        ],
        "answer": [1],
        "explanation": "Instance profiles deliver automatically rotated temporary credentials to the application through the instance metadata service. Long-term access keys on instances (or in user data) are a security risk, and root keys should never be used.",
    },
    {
        "id": 13, "domain": SECURE, "difficulty": "Hard",
        "question": "A three-tier application has web, application, and MySQL database tiers in a VPC. The database must accept connections only from the application tier and must not be reachable from the internet. Which TWO actions should a solutions architect take? (Choose TWO.)",
        "options": [
            "Place the database instances in private subnets that have no route to an internet gateway.",
            "Configure the database security group to allow inbound TCP 3306 only from the application tier's security group.",
            "Configure the database security group to allow inbound TCP 3306 from 0.0.0.0/0 and rely on a network ACL to deny internet traffic.",
            "Place the database in a public subnet and enable encryption at rest.",
            "Add an outbound deny rule to the application tier security group for all other destinations.",
        ],
        "answer": [0, 1],
        "explanation": "Private subnets remove internet reachability, and referencing the app tier security group as the source keeps the rule correct as app instances scale. Opening 0.0.0.0/0 is unsafe, encryption doesn't control network access, and security groups do not support deny rules.",
    },
    {
        "id": 14, "domain": SECURE, "difficulty": "Medium",
        "question": "A security team detects malicious traffic from a specific IP address range targeting EC2 instances in a public subnet. They must immediately block all traffic from that range to every instance in the subnet. What should a solutions architect do?",
        "options": [
            "Add a deny rule for the IP range to the security groups of the instances.",
            "Add an inbound deny rule for the IP range to the network ACL associated with the subnet, with a rule number lower than the allow rules.",
            "Remove the internet gateway from the VPC.",
            "Enable AWS Shield Standard on the instances.",
        ],
        "answer": [1],
        "explanation": "Network ACLs are stateless subnet-level filters that support explicit deny rules evaluated in rule-number order. Security groups only support allow rules, and removing the IGW would take down legitimate traffic.",
    },
    {
        "id": 15, "domain": SECURE, "difficulty": "Hard",
        "question": "A company with 40 AWS accounts in AWS Organizations wants employees to sign in once with their existing on-premises Microsoft Active Directory credentials and access multiple accounts with role-based permissions. The company wants to avoid synchronizing passwords to the cloud. What should a solutions architect recommend?",
        "options": [
            "Create IAM users in each account with the same user names as in Active Directory.",
            "Enable AWS IAM Identity Center, connect it to on-premises Active Directory by using AD Connector, and assign permission sets to groups for each account.",
            "Create an Amazon Cognito user pool federated with Active Directory and map users to IAM roles in each account.",
            "Configure a SAML identity provider in a single account and share IAM users through AWS RAM.",
        ],
        "answer": [1],
        "explanation": "IAM Identity Center provides centralized workforce SSO across Organizations accounts with permission sets. AD Connector proxies authentication to on-premises AD without caching or syncing passwords. Cognito is for customer-facing apps.",
    },
    {
        "id": 16, "domain": SECURE, "difficulty": "Medium",
        "question": "A company policy requires that every new Amazon EBS volume and snapshot created in an account is encrypted, even if developers forget to select encryption. What is the SIMPLEST way to enforce this?",
        "options": [
            "Create an AWS Config rule that deletes unencrypted volumes.",
            "Enable EBS encryption by default in each Region of the account.",
            "Use only instance store volumes.",
            "Write a Lambda function that encrypts volumes after they are created.",
        ],
        "answer": [1],
        "explanation": "EBS encryption by default is a per-Region account setting that automatically encrypts all new volumes and snapshot copies with the default or a chosen KMS key. Existing unencrypted volumes cannot be encrypted in place.",
    },
    {
        "id": 17, "domain": SECURE, "difficulty": "Hard",
        "question": "Account A owns an S3 bucket whose objects are encrypted with an SSE-KMS customer managed key in Account A. A role in Account B has been granted s3:GetObject in the bucket policy, but requests fail with AccessDenied. Which TWO changes are required? (Choose TWO.)",
        "options": [
            "Update the KMS key policy in Account A to allow the Account B role to use kms:Decrypt.",
            "Add kms:Decrypt for the Account A key ARN to the IAM policy attached to the role in Account B.",
            "Change bucket default encryption to SSE-S3 and re-upload all objects.",
            "Enable S3 Transfer Acceleration on the bucket.",
            "Create a KMS grant in Account B for an alias of the key.",
        ],
        "answer": [0, 1],
        "explanation": "Cross-account use of a KMS key requires permission on both sides: the key policy in the owning account must allow the external principal, and the external principal's IAM policy must allow the KMS action on that key ARN.",
    },
    {
        "id": 18, "domain": SECURE, "difficulty": "Medium",
        "question": "A third-party monitoring vendor needs read-only access to resources in a company's AWS account. The vendor serves many customers, and the company wants to prevent the 'confused deputy' problem. What should a solutions architect do?",
        "options": [
            "Create an IAM user for the vendor and share the access keys over an encrypted channel.",
            "Create an IAM role that trusts the vendor's AWS account and requires a unique external ID in the trust policy condition.",
            "Share the root user credentials protected by MFA.",
            "Create a resource-based policy on every resource that allows the vendor's account.",
        ],
        "answer": [1],
        "explanation": "A cross-account role with an sts:ExternalId condition ensures the vendor can assume the role only on behalf of this specific customer, mitigating the confused deputy problem.",
    },
    {
        "id": 19, "domain": SECURE, "difficulty": "Hard",
        "question": "A financial firm must retain trade records in Amazon S3 for 7 years in WORM (write once, read many) format. Regulators require that no user, including the root user, can delete or overwrite the objects during the retention period. Which solution meets this requirement?",
        "options": [
            "Enable S3 Versioning and MFA Delete on the bucket.",
            "Enable S3 Object Lock in governance mode with a 7-year retention period.",
            "Enable S3 Object Lock in compliance mode with a 7-year retention period.",
            "Use an S3 Lifecycle policy that transitions objects to S3 Glacier Deep Archive.",
        ],
        "answer": [2],
        "explanation": "Compliance mode prevents any user, including root, from deleting or shortening retention. Governance mode can be bypassed by users with s3:BypassGovernanceRetention. MFA Delete and lifecycle policies are not WORM controls.",
    },
    {
        "id": 20, "domain": SECURE, "difficulty": "Medium",
        "question": "An e-commerce company is concerned about large-scale DDoS attacks against its CloudFront and Route 53 hosted application. It needs 24/7 access to the AWS Shield Response Team and protection against scaling charges caused by DDoS attacks. Which service should it use?",
        "options": ["AWS Shield Standard", "AWS Shield Advanced", "AWS WAF only", "Amazon GuardDuty"],
        "answer": [1],
        "explanation": "Shield Advanced adds enhanced detection, 24/7 Shield Response Team access, and DDoS cost protection. Shield Standard is free and automatic but does not include these features.",
    },

    # ─────────────────────────────────────────────────────────────
    # DOMAIN 2 – DESIGN RESILIENT ARCHITECTURES (17)
    # ─────────────────────────────────────────────────────────────
    {
        "id": 21, "domain": RESILIENT, "difficulty": "Medium",
        "question": "An order-processing application has a web tier that receives orders and a processing tier on EC2. During flash sales, the processing tier is overwhelmed and orders are lost. How should a solutions architect redesign the application so that no orders are lost and processing scales with demand?",
        "options": [
            "Increase the instance size of the processing tier.",
            "Have the web tier write orders to an Amazon SQS queue and run the processing tier in an Auto Scaling group that scales on queue depth.",
            "Place the processing tier behind a Network Load Balancer.",
            "Use Amazon SNS to send orders directly to the processing instances over HTTP.",
        ],
        "answer": [1],
        "explanation": "SQS durably buffers orders and decouples the tiers; scaling consumers on queue metrics (e.g., backlog per instance) matches capacity to load. SNS-to-HTTP push does not buffer if subscribers are overwhelmed.",
    },
    {
        "id": 22, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A banking application sends account transactions through a message queue. Transactions for each account must be processed in the exact order they were sent, and duplicates must not be processed. Which solution meets these requirements?",
        "options": [
            "Amazon SQS standard queue with long polling.",
            "Amazon SQS FIFO queue using the account ID as the message group ID and content-based deduplication.",
            "Amazon SNS standard topic with an SQS standard queue subscription.",
            "Amazon Kinesis Data Firehose delivering to Amazon S3.",
        ],
        "answer": [1],
        "explanation": "FIFO queues preserve order within a message group and provide exactly-once processing within the deduplication interval. Using the account ID as the group ID allows parallelism across accounts while keeping per-account order.",
    },
    {
        "id": 23, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A company runs a production Amazon RDS for MySQL database. It requires automatic failover with minimal downtime if the primary database instance or its Availability Zone fails. What should a solutions architect configure?",
        "options": [
            "Create a read replica in another Availability Zone and promote it manually on failure.",
            "Convert the DB instance to a Multi-AZ deployment.",
            "Take automated snapshots every 5 minutes.",
            "Create a cross-Region read replica.",
        ],
        "answer": [1],
        "explanation": "RDS Multi-AZ maintains a synchronous standby in another AZ and fails over automatically by updating the DNS endpoint. Read replicas are asynchronous and require promotion; snapshots require a restore.",
    },
    {
        "id": 24, "domain": RESILIENT, "difficulty": "Hard",
        "question": "A company needs a disaster recovery strategy for a critical application in another AWS Region. Requirements: RPO of a few minutes, RTO under 1 hour, and the LOWEST possible cost. The database must be continuously replicated, but application servers may be launched only during a disaster. Which DR strategy fits BEST?",
        "options": ["Backup and restore", "Pilot light", "Warm standby", "Multi-site active/active"],
        "answer": [1],
        "explanation": "Pilot light keeps core data (the database) continuously replicated while other resources are provisioned only during recovery, giving minutes of RPO and an RTO in tens of minutes at low cost. Warm standby runs a scaled-down full stack (higher cost); backup and restore has hours of RTO.",
    },
    {
        "id": 25, "domain": RESILIENT, "difficulty": "Hard",
        "question": "A global application uses Amazon Aurora MySQL. The business requires the database to survive a full Regional outage with an RPO of about 1 second and an RTO of under 1 minute, while also serving low-latency reads in a second Region. Which solution meets these requirements?",
        "options": [
            "Aurora Multi-AZ cluster with cross-Region automated snapshot copies.",
            "Amazon Aurora Global Database with a secondary cluster in the second Region.",
            "Aurora cross-Region read replica created with binlog replication.",
            "AWS Database Migration Service continuous replication to an RDS instance in the second Region.",
        ],
        "answer": [1],
        "explanation": "Aurora Global Database uses storage-based replication with typical lag under 1 second and supports managed cross-Region failover in about a minute, while secondary clusters serve local reads.",
    },
    {
        "id": 26, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A company has a primary web application in us-east-1 and a static maintenance site in an S3 bucket in us-west-2. Users must be sent to the maintenance site automatically only if the primary application becomes unhealthy. Which Route 53 configuration should be used?",
        "options": [
            "Weighted routing with 100/0 weights.",
            "Latency-based routing.",
            "Failover routing with a health check on the primary record and the S3 website as the secondary record.",
            "Geolocation routing.",
        ],
        "answer": [2],
        "explanation": "Failover routing supports active-passive configurations: Route 53 returns the secondary record only when the primary's health check fails.",
    },
    {
        "id": 27, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A web application runs on EC2 instances in an Auto Scaling group behind an ALB. Users are logged out whenever a scale-in event terminates an instance because session data is stored in instance memory. How can a solutions architect make the application resilient to scaling events?",
        "options": [
            "Enable sticky sessions on the ALB.",
            "Store session state in Amazon ElastiCache for Redis or Amazon DynamoDB.",
            "Disable scale-in on the Auto Scaling group.",
            "Store session state on an attached EBS volume.",
        ],
        "answer": [1],
        "explanation": "Externalizing session state makes the web tier stateless so any instance can serve any request. Sticky sessions still lose state when the instance terminates, and EBS volumes are tied to a single instance.",
    },
    {
        "id": 28, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A VPC has private subnets in three Availability Zones that use a single NAT gateway in AZ-a for outbound internet access. The company wants outbound connectivity to continue if any single AZ fails. What should a solutions architect do?",
        "options": [
            "Replace the NAT gateway with a larger NAT instance.",
            "Create a NAT gateway in a public subnet in each AZ and configure each private subnet's route table to use the NAT gateway in its own AZ.",
            "Add a second internet gateway to the VPC.",
            "Enable cross-zone load balancing on the NAT gateway.",
        ],
        "answer": [1],
        "explanation": "NAT gateways are redundant only within their own AZ. Deploying one per AZ with AZ-specific routing removes the cross-AZ single point of failure (and cross-AZ data charges). A VPC supports only one internet gateway.",
    },
    {
        "id": 29, "domain": RESILIENT, "difficulty": "Hard",
        "question": "A company must replicate all new objects from an S3 bucket in eu-west-1 to a bucket in eu-central-1 for disaster recovery. Which TWO steps are required? (Choose TWO.)",
        "options": [
            "Enable S3 Versioning on both the source and destination buckets.",
            "Create a replication rule on the source bucket with an IAM role that S3 can assume to replicate objects.",
            "Enable S3 Transfer Acceleration on the destination bucket.",
            "Enable S3 Object Lock on the source bucket.",
            "Configure an S3 event notification to copy objects with AWS Lambda.",
        ],
        "answer": [0, 1],
        "explanation": "S3 Cross-Region Replication requires versioning on both buckets and a replication configuration with an IAM role granting S3 permission to read from the source and replicate to the destination. Existing objects need S3 Batch Replication.",
    },
    {
        "id": 30, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A Lambda function processes messages from an SQS queue. A few malformed messages fail repeatedly, are retried indefinitely, and block processing capacity. How should a solutions architect handle these messages?",
        "options": [
            "Increase the Lambda function timeout.",
            "Configure a redrive policy on the queue with a dead-letter queue and an appropriate maxReceiveCount.",
            "Reduce the queue's message retention period to 1 minute.",
            "Switch the queue to a FIFO queue.",
        ],
        "answer": [1],
        "explanation": "A dead-letter queue isolates messages that exceed maxReceiveCount so they can be analyzed without blocking the main queue. Shortening retention would lose valid messages as well.",
    },
    {
        "id": 31, "domain": RESILIENT, "difficulty": "Medium",
        "question": "When an order is placed, three independent microservices (billing, shipping, analytics) must each receive and process a copy of the event at their own pace. A failure in one service must not affect the others. Which architecture should be used?",
        "options": [
            "Send the event to a single SQS queue polled by all three services.",
            "Publish the event to an Amazon SNS topic with a separate SQS queue subscribed for each service (fan-out).",
            "Have the order service call each microservice synchronously through API Gateway.",
            "Write the event to an EBS volume shared by the three services.",
        ],
        "answer": [1],
        "explanation": "SNS-to-SQS fan-out delivers a copy of each message to every queue; each service consumes independently with its own buffering and retries. A single queue would deliver each message to only one consumer.",
    },
    {
        "id": 32, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A content management system runs on Linux EC2 instances across three Availability Zones. All instances must read and write the same set of files concurrently, and the storage must remain available if an AZ fails. Which storage solution should be used?",
        "options": [
            "Amazon EBS io2 volumes with Multi-Attach.",
            "Amazon EFS file system with mount targets in each AZ.",
            "Instance store volumes synchronized with rsync.",
            "Amazon S3 mounted as a block device.",
        ],
        "answer": [1],
        "explanation": "Amazon EFS is a managed NFS file system that stores data redundantly across multiple AZs and supports concurrent access from thousands of instances. EBS Multi-Attach is limited to instances in a single AZ.",
    },
    {
        "id": 33, "domain": RESILIENT, "difficulty": "Medium",
        "question": "An Auto Scaling group behind an ALB uses EC2 status checks. Occasionally the application process hangs while the instance remains running, and the instance continues to receive traffic and return errors. How can the solutions architect ensure such instances are automatically replaced?",
        "options": [
            "Enable detailed CloudWatch monitoring.",
            "Configure the Auto Scaling group to use Elastic Load Balancing health checks in addition to EC2 status checks.",
            "Increase the ALB idle timeout.",
            "Enable termination protection on the instances.",
        ],
        "answer": [1],
        "explanation": "With ELB health checks enabled, the ASG treats an instance that fails the target group's application-level health check as unhealthy and replaces it. EC2 status checks only detect infrastructure/OS-level problems.",
    },
    {
        "id": 34, "domain": RESILIENT, "difficulty": "Hard",
        "question": "A company connects its data center to AWS with a single 10 Gbps AWS Direct Connect connection. Management wants a backup path in case the Direct Connect connection fails, at the LOWEST cost, and accepts reduced bandwidth during the failure. What should a solutions architect recommend?",
        "options": [
            "Provision a second 10 Gbps Direct Connect connection at a different location.",
            "Configure an AWS Site-to-Site VPN connection to the same virtual private gateway or transit gateway as a backup path.",
            "Use VPC peering between the data center and the VPC.",
            "Use S3 Transfer Acceleration for all traffic.",
        ],
        "answer": [1],
        "explanation": "A Site-to-Site VPN over the internet is a low-cost backup for Direct Connect; BGP prefers the Direct Connect path and fails over to VPN automatically. A second DX gives maximum resiliency but at higher cost.",
    },
    {
        "id": 35, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A gaming company needs a NoSQL database that supports active-active writes in three AWS Regions with low-latency local reads and writes for players worldwide. Which solution should a solutions architect choose?",
        "options": [
            "Amazon RDS for PostgreSQL with cross-Region read replicas.",
            "Amazon DynamoDB global tables.",
            "Amazon ElastiCache for Redis with cluster mode enabled.",
            "Amazon DocumentDB with a single-Region cluster.",
        ],
        "answer": [1],
        "explanation": "DynamoDB global tables provide fully managed, multi-Region, multi-active replication so applications can read and write locally in each Region. RDS read replicas are read-only.",
    },
    {
        "id": 36, "domain": RESILIENT, "difficulty": "Medium",
        "question": "A loan approval workflow involves several Lambda functions that must run in sequence, with parallel branches, retries with exponential backoff, and a human approval step that may take days. Which service should coordinate the workflow with the LEAST custom code?",
        "options": [
            "Amazon SQS with a separate queue between each step.",
            "AWS Step Functions Standard Workflows.",
            "Amazon EventBridge Scheduler.",
            "A coordinator Lambda function that calls the other functions synchronously.",
        ],
        "answer": [1],
        "explanation": "Step Functions Standard Workflows provide built-in sequencing, parallel states, retry/catch logic and long-running executions (up to 1 year) with callback task tokens for human approval. A coordinator Lambda would be limited to 15 minutes.",
    },
    {
        "id": 37, "domain": RESILIENT, "difficulty": "Hard",
        "question": "An application must have at least 4 EC2 instances running at all times, even if an entire Availability Zone fails. The Region has three AZs. Which deployment is the MOST cost-effective way to meet this requirement?",
        "options": [
            "4 instances across 2 AZs (2 per AZ).",
            "6 instances across 2 AZs (3 per AZ).",
            "6 instances across 3 AZs (2 per AZ).",
            "8 instances across 2 AZs (4 per AZ).",
        ],
        "answer": [2],
        "explanation": "With 2 instances in each of 3 AZs, losing any AZ leaves 4 instances. 4 across 2 AZs leaves 2, 6 across 2 AZs leaves 3, and 8 across 2 AZs works but costs more.",
    },

    # ─────────────────────────────────────────────────────────────
    # DOMAIN 3 – DESIGN HIGH-PERFORMING ARCHITECTURES (15)
    # ─────────────────────────────────────────────────────────────
    {
        "id": 38, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "A reporting team runs heavy read queries against a production Amazon RDS for PostgreSQL database, slowing down the transactional application. What is the MOST effective way to improve application performance with minimal changes?",
        "options": [
            "Enable Multi-AZ and point reports to the standby instance.",
            "Create an RDS read replica and direct the reporting queries to the replica endpoint.",
            "Increase the allocated storage of the primary instance.",
            "Migrate the database to Amazon DynamoDB.",
        ],
        "answer": [1],
        "explanation": "Read replicas offload read-only workloads from the primary. In a standard Multi-AZ instance deployment, the standby cannot serve read traffic.",
    },
    {
        "id": 39, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "A retail application uses Amazon DynamoDB. During sales events, the same product items are read millions of times per minute and the application requires microsecond read latency. Which solution provides this with the LEAST application changes?",
        "options": [
            "Increase the table's provisioned read capacity units.",
            "Add Amazon DynamoDB Accelerator (DAX) in front of the table.",
            "Add a global secondary index.",
            "Enable DynamoDB Streams.",
        ],
        "answer": [1],
        "explanation": "DAX is an in-memory cache compatible with the DynamoDB API that delivers microsecond read latency for eventually consistent reads with minimal code change. More RCUs still yield millisecond latency.",
    },
    {
        "id": 40, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "A media company hosts images and videos in an S3 bucket in us-east-1. Users in Asia and Europe report slow load times. What should a solutions architect do to improve performance for global users?",
        "options": [
            "Enable S3 Cross-Region Replication to every Region.",
            "Create an Amazon CloudFront distribution with the S3 bucket as the origin.",
            "Move the content to EBS volumes attached to larger EC2 instances.",
            "Enable S3 Transfer Acceleration for downloads.",
        ],
        "answer": [1],
        "explanation": "CloudFront caches content at edge locations close to users, reducing latency for downloads globally. Transfer Acceleration primarily speeds up long-distance uploads and is not a cache.",
    },
    {
        "id": 41, "domain": PERFORMANCE, "difficulty": "Hard",
        "question": "A multiplayer game uses UDP and runs on EC2 behind Network Load Balancers in three Regions. Players need the lowest possible latency, and the game client must use a small set of static IP addresses with fast Regional failover. Which service should be used?",
        "options": [
            "Amazon CloudFront",
            "AWS Global Accelerator",
            "Amazon Route 53 latency-based routing",
            "Application Load Balancer with cross-zone load balancing",
        ],
        "answer": [1],
        "explanation": "Global Accelerator provides two static anycast IPs, routes TCP/UDP traffic over the AWS global network to the nearest healthy endpoint, and fails over in seconds. CloudFront does not support arbitrary UDP traffic, and DNS-based routing is subject to client caching.",
    },
    {
        "id": 42, "domain": PERFORMANCE, "difficulty": "Hard",
        "question": "Users around the world upload video files of 5–20 GB to a single S3 bucket in us-west-2. Uploads are slow and often fail partway. Which TWO actions will improve upload speed and reliability? (Choose TWO.)",
        "options": [
            "Enable S3 Transfer Acceleration and use the accelerate endpoint.",
            "Use multipart upload so parts are uploaded in parallel and failed parts can be retried individually.",
            "Enable S3 Versioning on the bucket.",
            "Change the storage class to S3 One Zone-IA.",
            "Create a CloudFront distribution with caching enabled for PUT requests.",
        ],
        "answer": [0, 1],
        "explanation": "Transfer Acceleration routes uploads through edge locations over the AWS backbone, and multipart upload parallelizes and allows retrying only failed parts. Storage class and versioning don't affect upload performance.",
    },
    {
        "id": 43, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "A scientific computing application runs a tightly coupled MPI workload on many EC2 instances that require the lowest possible network latency and highest packet-per-second throughput between nodes. How should the instances be deployed?",
        "options": [
            "In a spread placement group across multiple AZs.",
            "In a cluster placement group in a single AZ, using instance types that support Elastic Fabric Adapter (EFA).",
            "In a partition placement group across three AZs.",
            "In an Auto Scaling group across all AZs in the Region.",
        ],
        "answer": [1],
        "explanation": "Cluster placement groups pack instances close together in one AZ for low-latency, high-throughput networking; EFA further accelerates MPI/HPC communication. Spread and partition groups are designed for fault isolation.",
    },
    {
        "id": 44, "domain": PERFORMANCE, "difficulty": "Hard",
        "question": "A mission-critical OLTP database on EC2 requires 150,000 sustained IOPS with sub-millisecond latency on a single volume. Which EBS volume type should be used?",
        "options": [
            "General Purpose SSD (gp3)",
            "Provisioned IOPS SSD (io2 Block Express)",
            "Throughput Optimized HDD (st1)",
            "Cold HDD (sc1)",
        ],
        "answer": [1],
        "explanation": "io2 Block Express supports up to 256,000 IOPS per volume with sub-millisecond latency. gp3 maxes out at 16,000 IOPS, and HDD volumes are optimized for throughput, not IOPS.",
    },
    {
        "id": 45, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "A company ingests clickstream data from millions of users. Several applications must process the same records in real time, and the data must be replayable for up to 7 days for reprocessing. Which service should be used?",
        "options": [
            "Amazon Data Firehose",
            "Amazon Kinesis Data Streams with extended retention",
            "Amazon SQS standard queue",
            "Amazon SNS",
        ],
        "answer": [1],
        "explanation": "Kinesis Data Streams supports multiple concurrent consumers (including enhanced fan-out), ordered records per shard, and retention up to 365 days for replay. Firehose is a delivery service without replay; SQS messages are deleted once consumed.",
    },
    {
        "id": 46, "domain": PERFORMANCE, "difficulty": "Hard",
        "question": "A data team queries 50 TB of CSV log files in S3 with Amazon Athena. Most queries filter by date and select only a few columns, but they are slow and expensive. What will MOST improve performance and reduce cost?",
        "options": [
            "Move the data to S3 Standard-IA.",
            "Convert the data to a columnar format such as Apache Parquet with compression and partition it by date.",
            "Increase the Athena query timeout.",
            "Load the data into Amazon RDS and query it with SQL.",
        ],
        "answer": [1],
        "explanation": "Athena charges by data scanned. Columnar formats let queries read only the needed columns, compression reduces bytes, and date partitions let Athena prune irrelevant data.",
    },
    {
        "id": 47, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "A latency-sensitive API built on API Gateway and AWS Lambda has unacceptable response times for the first requests after periods of inactivity due to cold starts. What should a solutions architect do?",
        "options": [
            "Increase the Lambda function timeout.",
            "Configure provisioned concurrency for the Lambda function alias.",
            "Configure reserved concurrency of 1.",
            "Move the function to a VPC.",
        ],
        "answer": [1],
        "explanation": "Provisioned concurrency keeps a specified number of execution environments initialized and ready, eliminating cold-start latency. Reserved concurrency only caps/guarantees concurrency; it doesn't pre-warm.",
    },
    {
        "id": 48, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "A web application's traffic spikes every weekday at 9:00 AM. New EC2 instances take about 10 minutes to become ready, so users experience slow responses every morning while dynamic scaling catches up. What should a solutions architect do?",
        "options": [
            "Use a simple scaling policy with a shorter cooldown.",
            "Create a scheduled scaling action that increases the Auto Scaling group's desired capacity shortly before 9:00 AM each weekday.",
            "Use a larger instance type.",
            "Enable detailed monitoring for 1-minute metrics.",
        ],
        "answer": [1],
        "explanation": "For known, recurring load patterns, scheduled scaling provisions capacity before demand arrives. Predictive scaling is another option; reactive policies always lag behind instance warm-up time.",
    },
    {
        "id": 49, "domain": PERFORMANCE, "difficulty": "Hard",
        "question": "A genomics company runs HPC jobs that need a high-performance shared file system with hundreds of GB/s throughput. The input datasets reside in Amazon S3, and results must be written back to S3. Which storage service should be used?",
        "options": [
            "Amazon EFS with Elastic Throughput",
            "Amazon FSx for Lustre linked to the S3 bucket",
            "Amazon FSx for Windows File Server",
            "Amazon EBS gp3 volumes",
        ],
        "answer": [1],
        "explanation": "FSx for Lustre is built for HPC, scales to hundreds of GB/s, and natively integrates with S3 (lazy-loading data and exporting results via data repository associations).",
    },
    {
        "id": 50, "domain": PERFORMANCE, "difficulty": "Hard",
        "question": "Worker instances in an Auto Scaling group process messages from an SQS queue. Each instance can process about 10 messages per second, and the business needs each message processed within 1 minute of arrival. Which scaling approach BEST matches capacity to the workload?",
        "options": [
            "Target tracking on average CPU utilization of 50%.",
            "Target tracking on a custom 'backlog per instance' metric (ApproximateNumberOfMessagesVisible ÷ running instances) with a target of about 600.",
            "Scheduled scaling every hour.",
            "Step scaling on the NumberOfMessagesDeleted metric.",
        ],
        "answer": [1],
        "explanation": "Backlog per instance reflects the work each instance must do. With 10 msg/s and a 60-second latency target, each instance can handle a backlog of ~600 messages. CPU doesn't reliably reflect queue depth.",
    },
    {
        "id": 51, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "A social networking application must run queries such as 'find friends of friends who like the same bands' across billions of highly connected relationships with low latency. Which database is the BEST fit?",
        "options": ["Amazon RDS for MySQL", "Amazon Neptune", "Amazon Redshift", "Amazon DynamoDB"],
        "answer": [1],
        "explanation": "Neptune is a purpose-built graph database optimized for traversing highly connected data with Gremlin, openCypher or SPARQL. Relational joins and key-value lookups perform poorly for deep relationship traversal.",
    },
    {
        "id": 52, "domain": PERFORMANCE, "difficulty": "Medium",
        "question": "An application uses an Amazon Aurora MySQL cluster with one writer and three Aurora Replicas. Read traffic is unevenly distributed because the application connects to a specific replica endpoint. How should the application distribute read traffic and automatically include replicas added later?",
        "options": [
            "Use the cluster (writer) endpoint for all queries.",
            "Use the Aurora reader endpoint for read queries and enable Aurora Auto Scaling for replicas.",
            "Create a Network Load Balancer in front of the replicas.",
            "Use Route 53 weighted records for each replica instance endpoint.",
        ],
        "answer": [1],
        "explanation": "The reader endpoint load-balances connections across all available Aurora Replicas, including ones added by Aurora Auto Scaling, with no application changes.",
    },

    # ─────────────────────────────────────────────────────────────
    # DOMAIN 4 – DESIGN COST-OPTIMIZED ARCHITECTURES (13)
    # ─────────────────────────────────────────────────────────────
    {
        "id": 53, "domain": COST, "difficulty": "Medium",
        "question": "A company stores millions of objects in S3 with unpredictable and changing access patterns. Some objects are accessed frequently for a few weeks and then rarely; others are never accessed. The company wants to minimize storage costs without operational overhead or retrieval fees. Which storage class should be used?",
        "options": ["S3 Standard", "S3 Intelligent-Tiering", "S3 Standard-IA", "S3 One Zone-IA"],
        "answer": [1],
        "explanation": "Intelligent-Tiering automatically moves objects between access tiers based on usage, with no retrieval fees, making it ideal for unknown or changing access patterns.",
    },
    {
        "id": 54, "domain": COST, "difficulty": "Hard",
        "question": "Application logs are accessed frequently for 30 days, then almost never. They must be retained for 7 years for audits, and retrieval within 48 hours is acceptable after the first 30 days. Which solution is MOST cost-effective?",
        "options": [
            "Keep logs in S3 Standard for 7 years.",
            "Use an S3 Lifecycle rule to transition objects to S3 Glacier Deep Archive after 30 days and expire them after 7 years.",
            "Use an S3 Lifecycle rule to transition objects to S3 Standard-IA after 30 days and expire them after 7 years.",
            "Store logs on EBS sc1 volumes and take yearly snapshots.",
        ],
        "answer": [1],
        "explanation": "Glacier Deep Archive is the lowest-cost storage class, with standard retrieval within 12 hours and bulk within 48 hours, which satisfies the requirement. Standard-IA costs significantly more for long-term retention.",
    },
    {
        "id": 55, "domain": COST, "difficulty": "Hard",
        "question": "A company runs steady 24/7 compute workloads and commits to them for 3 years. It expects to move some workloads from EC2 to AWS Fargate and Lambda, change instance families, and possibly change Regions. Which purchasing option provides significant savings with the MOST flexibility?",
        "options": [
            "Standard Reserved Instances",
            "EC2 Instance Savings Plans",
            "Compute Savings Plans",
            "Spot Instances",
        ],
        "answer": [2],
        "explanation": "Compute Savings Plans apply across instance families, sizes, Regions, OS, and tenancy, and also cover Fargate and Lambda. EC2 Instance Savings Plans are locked to an instance family in a Region; Standard RIs are even less flexible.",
    },
    {
        "id": 56, "domain": COST, "difficulty": "Medium",
        "question": "A company runs nightly batch image-processing jobs that are stateless, can be checkpointed, and can tolerate interruption. Jobs may finish at any time before 6:00 AM. Which EC2 purchasing option is MOST cost-effective?",
        "options": ["On-Demand Instances", "Spot Instances", "Dedicated Hosts", "Standard Reserved Instances"],
        "answer": [1],
        "explanation": "Spot Instances offer up to 90% discounts for interruption-tolerant, flexible workloads. Reserved Instances are suited to steady-state usage, not a few hours per night.",
    },
    {
        "id": 57, "domain": COST, "difficulty": "Hard",
        "question": "A company's AWS bill shows very high NAT gateway data processing charges. Investigation shows that most of the traffic is from EC2 instances in private subnets downloading data from Amazon S3 and writing to DynamoDB in the same Region. What is the MOST cost-effective way to reduce these charges?",
        "options": [
            "Replace the NAT gateway with a NAT instance.",
            "Create gateway VPC endpoints for Amazon S3 and Amazon DynamoDB and update the private subnet route tables.",
            "Create interface VPC endpoints for S3 and DynamoDB in every AZ.",
            "Move the instances to public subnets.",
        ],
        "answer": [1],
        "explanation": "Gateway endpoints for S3 and DynamoDB have no hourly or data processing charges and remove this traffic from the NAT gateway. Interface endpoints incur hourly and per-GB charges; public subnets weaken security.",
    },
    {
        "id": 58, "domain": COST, "difficulty": "Medium",
        "question": "Development and test environments run on On-Demand EC2 instances and RDS databases 24/7, but they are used only from 8:00 AM to 6:00 PM on weekdays. What is the MOST cost-effective way to reduce costs with minimal effort?",
        "options": [
            "Purchase 3-year Reserved Instances for all dev/test resources.",
            "Use the AWS Instance Scheduler (or EventBridge Scheduler with Systems Manager Automation) to stop the instances and databases outside business hours.",
            "Convert all instances to Dedicated Instances.",
            "Move the environments to a larger instance type that runs faster.",
        ],
        "answer": [1],
        "explanation": "Running 50 of 168 hours per week cuts compute costs by roughly 70%. Reserved Instances would bill for all hours, wasting the commitment on idle time.",
    },
    {
        "id": 59, "domain": COST, "difficulty": "Medium",
        "question": "A company suspects many of its EC2 instances, EBS volumes, and Lambda functions are over-provisioned. It wants ML-based right-sizing recommendations based on historical utilization. Which service should it use?",
        "options": ["AWS Trusted Advisor only", "AWS Compute Optimizer", "AWS Cost Explorer forecasting", "AWS Budgets"],
        "answer": [1],
        "explanation": "Compute Optimizer analyzes CloudWatch utilization metrics with ML to recommend optimal EC2 instance types, EBS volume configurations, Lambda memory sizes, and more.",
    },
    {
        "id": 60, "domain": COST, "difficulty": "Hard",
        "question": "A company must store medical images that are accessed about once per quarter. When requested, an image must be returned in milliseconds. Data must be durable across multiple AZs. Which storage class is MOST cost-effective?",
        "options": [
            "S3 Glacier Flexible Retrieval",
            "S3 Glacier Instant Retrieval",
            "S3 One Zone-IA",
            "S3 Standard",
        ],
        "answer": [1],
        "explanation": "Glacier Instant Retrieval offers millisecond retrieval at a lower storage price than Standard-IA for data accessed about once a quarter, with multi-AZ durability. Flexible Retrieval takes minutes to hours; One Zone-IA stores data in a single AZ.",
    },
    {
        "id": 61, "domain": COST, "difficulty": "Medium",
        "question": "A new DynamoDB-backed application has unpredictable, spiky traffic with long idle periods. The team cannot forecast capacity and wants to pay only for the requests it actually serves. Which capacity mode should be used?",
        "options": [
            "Provisioned capacity with a high fixed RCU/WCU setting",
            "On-demand capacity mode",
            "Provisioned capacity with reserved capacity purchased for 3 years",
            "DynamoDB Accelerator (DAX) only",
        ],
        "answer": [1],
        "explanation": "On-demand mode bills per request and instantly accommodates traffic spikes, which suits unpredictable workloads. Provisioned capacity at a high fixed level wastes money during idle periods.",
    },
    {
        "id": 62, "domain": COST, "difficulty": "Hard",
        "question": "A company runs containerized microservices on Amazon ECS using a cluster of EC2 instances that average 25% utilization. The operations team wants to stop managing servers and pay only for the resources each task requests. Some tasks are interruption-tolerant background jobs. Which solution is MOST cost-effective?",
        "options": [
            "Move all tasks to larger EC2 instances to improve bin packing.",
            "Migrate the services to AWS Fargate and run interruption-tolerant tasks with a Fargate Spot capacity provider.",
            "Migrate the services to Amazon EKS on EC2 On-Demand instances.",
            "Purchase Reserved Instances for the existing ECS cluster.",
        ],
        "answer": [1],
        "explanation": "Fargate removes server management and charges per task vCPU/memory, eliminating idle cluster capacity. Fargate Spot gives up to ~70% discount for interruption-tolerant tasks.",
    },
    {
        "id": 63, "domain": COST, "difficulty": "Medium",
        "question": "A finance team wants to receive an email alert when the projected monthly AWS spend for a project is forecast to exceed $10,000, before the money is actually spent. Which solution meets this requirement?",
        "options": [
            "A CloudWatch alarm on the EstimatedCharges metric only.",
            "An AWS Budgets cost budget filtered by the project's cost allocation tag with an alert on forecasted spend.",
            "AWS Cost and Usage Report delivered to S3.",
            "AWS Trusted Advisor cost optimization checks.",
        ],
        "answer": [1],
        "explanation": "AWS Budgets supports alerts on forecasted costs and can filter by cost allocation tags. CloudWatch billing alarms trigger on actual estimated charges, not forecasts, and can't easily filter by tag.",
    },
    {
        "id": 64, "domain": COST, "difficulty": "Hard",
        "question": "A web application needs a baseline of 10 instances at all times and bursts up to 50 instances during unpredictable peaks. The application is stateless and can handle instance interruptions gracefully. Which Auto Scaling configuration is MOST cost-effective?",
        "options": [
            "50 On-Demand instances running at all times.",
            "An Auto Scaling group with a mixed instances policy: On-Demand base capacity of 10 (covered by a Savings Plan) and 100% Spot Instances above the base, across multiple instance types.",
            "An Auto Scaling group using only Spot Instances of a single instance type.",
            "An Auto Scaling group using only On-Demand instances with scheduled scaling.",
        ],
        "answer": [1],
        "explanation": "Covering the steady baseline with discounted On-Demand capacity and serving bursts with diversified Spot capacity minimizes cost while maintaining availability. A single Spot instance type increases interruption risk for the baseline.",
    },
    {
        "id": 65, "domain": COST, "difficulty": "Medium",
        "question": "A company has hundreds of gp2 EBS volumes. Analysis shows that most volumes need only 3,000 IOPS and 125 MiB/s throughput. How can the company reduce storage costs without reducing performance?",
        "options": [
            "Migrate the volumes to st1 HDD volumes.",
            "Modify the volumes to gp3 using Elastic Volumes.",
            "Migrate the volumes to io2.",
            "Take snapshots and delete the volumes.",
        ],
        "answer": [1],
        "explanation": "gp3 is about 20% cheaper per GB than gp2 and includes a baseline of 3,000 IOPS and 125 MiB/s regardless of size. Elastic Volumes allows the change online without downtime.",
    },
]

assert len(QUESTIONS) == 65, f"Expected 65 questions, found {len(QUESTIONS)}"
assert len({q["id"] for q in QUESTIONS}) == 65, "Question ids must be unique"
