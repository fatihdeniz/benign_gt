Ip dataset consists of 5 groups:
Group 1: Malicious IPs (BLAG), 
Group 2: Benign CDN IPs (akamai and cloudflare), 
Group 3: Benign Cloud IPs (AWS, Azure and Google GCE, Rackspace), 
Group 4: Benign Large corporate IPs (Microsoft (exclude Azure), Amazon (Exclude AWS), Google (Exclude GCE), Facebook)
Group 5: Sampled from daily PDNS data (26/08/2022) - Assumed to be unbiased

Group 1 is the BLAG dataset, and Groups 2-3-4 are benign datasets. They have no intersections with Group 1. Group 5 is an unbiased (assumed so) pdns dataset from the same day. It has a small intersection (101 IPs) with Group 1.
These are the sizes of each group:
11072 group1.csv
16394 group2.csv
12252 group3.csv
 6394 group4.csv
25910 group5.csv

These are the features:

Average of hosted domains hosting features:
'feat_query_count', 'feat_ip_count', 'feat_ns_count', 'feat_isns_domain', 'feat_soadomain_count', 'feat_issoa_domain', 'feat_duration',

Hosting features of IPs:
'feat_ip_duration', 'feat_ip_query', 'feat_ip_apexcount',

Whois features:
'number of whois records', 'number of inetnums', 'size of the maximum intentum', 'size of the minimum intentum', 'netTypes', 'number of owners', 'most recent update',

VT features:
'#resolutions', '#res_apexes', 'detected_urls', 'detected_urls_pos', 'detected_apexes', 'detected_fqdns', 'undetected_urls', 'undetected_apexes', 'undetected_fqdns', 'det_comm_files', 'undet_comm_files', 'det_download', ' undet_download', ' det_referrer', ' undet_referrer',

Encoded subnet asn:
'subnet', 'asn'