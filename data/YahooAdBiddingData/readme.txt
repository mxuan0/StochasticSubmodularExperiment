Dataset: ydata-ysm-advertiser-bids-v1_0

Yahoo! Search Marketing advertiser bidding data, version 1.0

=====================================================================
This dataset is provided as part of the Yahoo! Research Alliance
Webscope program, to be used for approved non-commercial research
purposes by recipients who have signed a Data Sharing Agreement with
Yahoo!. This dataset is not to be redistributed. No personally
identifying information is available in this dataset. More information
about Yahoo! Research Webscope is available at
http://research.yahoo.com
=====================================================================

Full description:

This dataset includes Yahoo! Search Marketing advertiser bid data in
the following format:

FILE:           ( LINE '\n' ) +
LINE:           TIMESTAMP '\t' PHRASE_ID '\t' ACCOUNT_ID '\t' PRICE '\t' AUTO
TIMESTAMP:      MM/DD/YYYY HH:MM:SS
PHRASE_ID:      Int
ACCOUNT_ID:     Int
PRICE:          Float
AUTO:           0 or 1

For the Auto field, 0 means that the bid was placed manually, 1 that
the bid was placed by an automatic bidding program.  Bids are given
for fifteen minute increments.  The bidded phrases/accounts used to
generate this dataset are the top 1,000 phrases by volume and all
associated accounts, for the time period from 6/15/2002 to
6/14/2003.

Price is denominated in US dollars.

Data snippet:

06/15/2002 00:00:00     39      691     1.34    0
06/15/2002 00:00:00     40      691     1.16    0
06/15/2002 00:00:00     83      691     .85     0
06/15/2002 00:00:00     1       741     13.71   0
06/15/2002 00:00:00     1       741     13.73   0
