Columns (33)

Column Name
Description

API Field Name

Data Type
UniqueID
Unique formula: @Value(ttxid)-@Value(certificate_number)
uniqueid
Text
Business Account Number
Seven digit number assigned to registered business accounts
certificate_number
Text
Location Id
Location identifier
ttxid
Text
Ownership Name
Business owner(s) name
ownership_name
Text
DBA Name
Doing Business As Name or Location Name
dba_name
Text
Street Address
Business location street address
full_business_address
Text
City
Business location city
city
Text
State
Business location state
state
Text
Source Zipcode
Business location zip code
business_zip
Text
Business Start Date
Start date of the business
dba_start_date
Floating Timestamp
Business End Date
End date of the business
dba_end_date
Floating Timestamp
Location Start Date
Start date at the location
location_start_date
Floating Timestamp
Location End Date
End date at the location, if closed
location_end_date
Floating Timestamp
Administratively Closed
Business locations marked as “Administratively Closed” have not filed or communicated with TTX for 3 years, or were marked as closed following a notification from another City and County Department.
administratively_closed
Text
Mail Address
Address for mailing
mailing_address_1
Text
Mail City
Mailing address city
mail_city
Text
Mail State
Mailing address state
mail_state
Text
Mail Zipcode
Mailing address zipcode
mail_zipcode
Text
NAICS Code
The North American Industry Classification System (NAICS) is a standard used by Federal statistical agencies for the purpose of collecting, analyzing and publishing statistical data related to the U.S. business economy. A subset of these are options on the business registration form used in the administration of the City and County's tax code. The registrant indicates the business activity on the City and County's tax registration forms. See NAICS Codes tab in the attached data dictionary under About > Attachments.
naic_code
Text
NAICS Code Description
The Business Activity that the NAICS code maps on to ("Multiple" if there are multiple codes indicated for the business).
naic_code_description
Text
NAICS Code Descriptions List
A list of all NAICS code descriptions separated by semi-colon
naics_code_descriptions_list
Text
LIC Code
The LIC code of the business, if multiple, separated by spaces
lic
Text
LIC Code Description
The LIC code description ("Multiple" if there are multiple codes for a business)
lic_code_description
Text
LIC Code Descriptions List
A list of all LIC code descriptions separated by semi-colon
lic_code_descriptions_list
Text
Parking Tax
Whether or not this business pays the parking tax
parking_tax
Checkbox
Transient Occupancy Tax
Whether or not this business pays the transient occupancy tax
transient_occupancy_tax
Checkbox
Business Location
The latitude and longitude of the business location for mapping purposes.
location
Point
Business Corridor
The Business Corridor in which the the business location falls, if it is in one. Not all business locations are in a corridor. Boundary reference: https://data.sfgov.org/d/h7xa-2xwk
business_corridor
Text
Neighborhoods - Analysis Boundaries
The Analysis Neighborhood in which the business location falls. Not applicable outside of San Francisco. Boundary reference: https://data.sfgov.org/d/p5b7-5n3h
neighborhoods_analysis_boundaries
Text
Supervisor District
The Supervisor District in which the business location falls. Not applicable outside of San Francisco. Boundary reference: https://data.sfgov.org/d/xz9b-wyfc
supervisor_district
Text
Community Benefit District
The Community Benefit District in which the business location falls. Not applicable outside of San Francisco. Boundary reference: https://data.sfgov.org/d/c28a-f6gs
community_benefit_district
Text
data_as_of
Timestamp the data was updated in the source system
data_as_of
Floating Timestamp
data_loaded_at
Timestamp the data was loaded here (open data portal)
data_loaded_at
Floating Timestamp