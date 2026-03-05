Places Pro
Includes all fields available via our Places Open Source data and extra fields about chain membership and location. If no explicit fields are provided in an API request, all Pro fields will be returned by default.

Field	Description
fsq_place_id	A unique identifier for a FSQ Place (formerly known as Venue ID).
name	The best known name for the FSQ Place.
categories	
An array, possibly empty, of categories that describe the FSQ Place. Included subfields: id (Category ID), name (Category Label), and icon (Category's Icon).

View the list of FSQ Categories → Places Photos Guide

location	
An object containing the following fields, if available:

address
locality
region
postcode
country
admin_region
post_town
po_box
latitude	The latitude of the location's main entrance.
longitude	The longitude of the location's main entrance.
distance	The calculated distance (in meters) from the provided location (i.e. ll + radius OR near OR ne + sw) in the API call. This field will only be returned by the Place Search endpoint.
tel	The best known telephone number, with local formatting.
email	The primary contact email address for the FSQ Place.
website	The official website for the FSQ Place.
social_media	An object containing a FSQ Place's social media identifiers. Included subfields: facebook_id, instagram, and twitter. Not all FSQ Places will have all subfields.
link	The URL associated with the FSQ Place Detail API call.
date_closed	The recorded date when the FSQ Place was marked as permanently closed in Foursquare's databases. This does not necessarily indicate the Place was actually closed on this date.
placemaker_url	The Foursquare Placemaker URL associated with the FSQ Place
chains	
An array, possibly empty, of chains that the FSQ Place belongs to. Included subfields: id (Chain ID) and name (Chain Name).

View the list of FSQ Chains →

store_id	A unique ID assigned to a venue to differentiate it from other stores in the same chain.
related_places	
An object containing the information for Places related to the FSQ Place. Included subfields: parent and children. Not all FSQ Places will have parent or children relationships.

Some Children Place responses may not show a fsq_id. This indicates that Foursquare does not have enough data inputs to mark the Child Place as an official Place.

Example:
parent = Los Angeles International Airport
children = Terminal 1, Terminal 2, Terminal 3...

extended_location	An object containing additional location metadata, including census_block and dma.
unresolved_flags	
An array containing a set of quality issue flags reported by Placemakers, human or agent, that require further corroboration. Each flag marks a suspected problem that has not yet been resolved. The values can be one or more of the following:

closed – the place is believed to be permanently closed
duplicate – the place is a duplicate of another record
delete – the place should be removed entirely
privatevenue – the place is private and should not be publicly listed
inappropriate – the content violates policy or is otherwise unsuitable
doesnt_exist – the place is thought not to exist at the specified location
Places Premium
Includes all data from Places Pro and additional fields:

Field	Description
attributes	
A list of boolean tags that help to describe services and additional metadata offered by the Place (e.g. takes_reservations: true)

For a full list of tags, see Places Tags.

description	A general description of the FSQ Place. Typically provided by the owner/claimant of the FSQ Place and/or updated by City Guide Superusers.
hours	
An array containing the regular hours of operation, is_local_holiday, open_now, and a formatted display string.

regular hours contain subfields: day (1 = Monday, 2 = Tuesday, 7 = Sunday) and open/close (24-hour time).

hours_popular	
An array containing the hours during the week when a FSQ Place is most often visited. This place must have a minimum number of check-ins to be considered for the calculation.

Similar to hours, this field contains subfields: day (1 = Monday, 2 = Tuesday, 7 = Sunday) and open/close (24-hour time).

menu	A menu URL for the FSQ Place.
photos	
An array of objects containing the following subfields:

id
created_at
classifications
prefix
suffix
width
height.For more information, see our Places Photos Guide →
place_actions	A list of actionable links associated with the place, such as reservations or delivery ordering. Each action includes the action type, a URL for completing the action, and the provider source ID.
popularity	A measure of the FSQ Place's popularity, by foot traffic. This score is on a 0 to 1 scale and uses a 6-month span of Place visits for a given geographic area.
price	
A numerical value (from 1 to 4) that best describes the pricing tier of the FSQ Place, based on known prices for menu items and other offerings.

Values include:

1 = Cheap
2 = Moderate
3 = Expensive
4 = Very Expensive.
rating	A numerical rating (from 0.0 to 10.0) of the FSQ Place, based on user votes, likes/dislikes, tips sentiment, and visit data. Not all FSQ Places will have a rating.
stats	An object containing counts of photos, ratings, and tips. Included subfields: total_photos, total_ratings, and total_tips.
tastes	An array of up to 25 tastes describing the FSQ Place.
tips	An array containing the following subfields: created_at and text.
veracity_rating	
A 1 to 5 star quality rating indicating how complete, current, and reliable the location's information is, helping you understand how strong our data signals are for this place.

5 = Highly reliable and complete information, regularly updated from trusted sources and real user activity.
4 = Very good information verified from reliable sources with some user engagement.
3 = Good essential information from multiple sources that's generally current but less frequently updated.
2 = Basic information available but some details may be missing or outdated.
1 = Limited information from few sources that may not have been updated recently.

Places Core Data
Field	Description
fsq_id	A unique identifier for a FSQ Place (formerly known as Venue ID).
name	The best known name for the FSQ Place.
geocodes	
Set of geocodes (latitude, longitude) for the Place. The primary coordinates are identified by the main latitude and longitude.

Where available, other coordinates may include:

roof
drop_off
front_door
road
location	
An object containing none, some, or all of the following fields:

address
address_extended
locality
dma
region
postcode
country
admin_region
post_town
po_box
cross_street
formatted_address
census_block: populated only for Places within the United States, in all other countries it will be null.
NOTE: For the Autocomplete and Address Data APIs, the location object data is populated by the verified address data provided by Loqate.

View Loqate's Full Coverage List →

categories	
An array, possibly empty, of categories that describe the FSQ Place. Included subfields: id (Category ID), name (Category Label), and icon (Category's Icon).

View the list of FSQ Categories →

chains	
An array, possibly empty, of chains that the FSQ Place belongs to. Included subfields: id (Chain ID) and name (Chain Name).

View the list of FSQ Chains →

related_places	
An object containing the information for Places related to the FSQ Place. Included subfields: parent and children. Not all FSQ Places will have parent or children relationships.

Some Children Place responses may not show a fsq_id. This indicates that Foursquare does not have enough data inputs to mark the Child Place as an official Place.

Example:
parent = Los Angeles International Airport
children = Terminal 1, Terminal 2, Terminal 3...

timezone	A string that denotes the region representing the FSQ Place's timezone.
distance	The calculated distance (in meters) from the provided location (i.e. ll + radius OR near OR ne + sw) in the API call. This field will only be returned by the Place Search endpoint.
link	The URL associated with the corresponding Places Detail API call for the returned Place.
closed_bucket	
Probability that a POI is Open/Closed.

- VeryLikelyOpen, VeryLikelyClosed: 95% Confidence that a POI is Open/Closed respectively

- LikelyOpen, LikelyClosed: 80% Confidence that a POI identified as LikelyOpen/VeryLikelyOpen is Open . Similarly, 80% Confidence that a POI marked as LikelyClosed/VeryLikelyClosed is Closed

↑ back to top

Places Rich Data
Field	Description
description	A general description of the FSQ Place. Typically provided by the owner/claimant of the FSQ Place and/or updated by City Guide Superusers.
tel	The best known telephone number, with local formatting.
fax	The best known fax number, with local formatting.
email	The primary contact email address for the FSQ Place.
website	The official website for the FSQ Place.
social_media	An object containing a FSQ Place's social media identifiers. Included subfields: facebook_id, instagram, and twitter. Not all FSQ Places will have all subfields.
verified	A boolean that indicates whether or not the FSQ Place has been claimed.
hours	
An array containing the regular hours of operation, is_local_holiday, open_now, and a formatted display string.

regular hours contain subfields: day (1 = Monday, 2 = Tuesday, 7 = Sunday) and open/close (24-hour time).

hours_popular	
An array containing the hours during the week when a FSQ Place is most often visited. This place must have a minimum number of check-ins to be considered for the calculation.

Similar to hours, this field contains subfields: day (1 = Monday, 2 = Tuesday, 7 = Sunday) and open/close (24-hour time).

rating	
A numerical rating (from 0.0 to 10.0) of the FSQ Place, based on user votes, likes/dislikes, tips sentiment, and visit data. Not all FSQ Places will have a rating.

NOTE: v3 does not include the following mapped ratings to colors as could be found in v2:

0.0 = LightMediumGrey (#C7CDCF)
0.0 <> 4.0 = Red (#E6092C)
4.0 <> 5.0 = DarkOrange (#FF6701)
5.0 <> 6.0 = Orange (#FF9600)
6.0 <> 7.0 = Yellow (#FFC800)
7.0 <> 8.0 = LightGreen (#C5DE35)
8.0 <> 9.0 = Green (#73CF42)
9.0 + = DarkGreen (#00B551)
stats	An object containing counts of photos, ratings, and tips. Included subfields: total_photos, total_ratings, and total_tips.
popularity	A measure of the FSQ Place's popularity, by foot traffic. This score is on a 0 to 1 scale and uses a 6-month span of Place visits for a given geographic area.
price	
A numerical value (from 1 to 4) that best describes the pricing tier of the FSQ Place, based on known prices for menu items and other offerings.

Values include:

1 = Cheap
2 = Moderate
3 = Expensive
4 = Very Expensive.
menu	A linked menu for the FSQ Place.
date_closed	The recorded date when the FSQ Place was marked as permanently closed in Foursquare's databases. This does not necessarily indicate the Place was actually closed on this date.
photos	
An array of objects containing the following subfields:

id
created_at
classifications
prefix
suffix
width
height.
For more information, see our Photos Guide →

tips	An array containing the following subfields: created_at and text.
tastes	An array of up to 25 tastes that best describe the FSQ Place.
features	
A list of boolean tags that help to describe services and additional metadata offered by the Place (e.g. takes_reservations: true)

For a full list of tags, see Places Tags.

store_id	The unique ID assigned to a venue in order to differentiate it from other stores within the same chain.
venue_reality_bucket	This attribute represents how “real” Foursquare believes a venue to be. As venues can be submitted directly by users, Foursquare uses a proprietary algorithm to assess real venues (public places like a popular restaurant, store, concert venue, etc) vs a private or nonexistent venue. Foursquare’s Venue Reality algorithm uses a combination of explicit and implicit signals (examples include number of searches on Foursquare, number of photos/tips submitted, number of check-ins, etc) to bucket venues with an output of Low, Medium, High, VeryHigh
↑ back to top

Address Data
Field	Description
location	An object containing none, some, or all of the following fields: address, address_extended, locality, dma, region, postcode, country, admin_region, post_town, po_box, and cross_street.
geocodes	
Set of geocodes (latitude, longitude) for the Place. The primary coordinates are identified by the main latitude and longitude.

Where available, other coordinates may include: roof, drop_off, front_door, and road.


example response
{
  "results": [
    {
      "fsq_id": "string",
      "categories": [
        {
          "id": 0,
          "name": "string",
          "icon": {
            "id": "string",
            "created_at": "2023-10-11T16:32:04.439Z",
            "prefix": "string",
            "suffix": "string",
            "width": 0,
            "height": 0,
            "classifications": [
              "string"
            ],
            "tip": {
              "id": "string",
              "created_at": "2023-10-11T16:32:04.439Z",
              "text": "string",
              "url": "string",
              "lang": "string",
              "agree_count": 0,
              "disagree_count": 0
            }
          }
        }
      ],
      "chains": [
        {
          "id": "string",
          "name": "string"
        }
      ],
      "closed_bucket": "string",
      "date_closed": "2023-10-11",
      "description": "string",
      "distance": 0,
      "email": "string",
      "fax": "string",
      "features": {
        "payment": {
          "credit_cards": {
            "accepts_credit_cards": {},
            "amex": {},
            "discover": {},
            "visa": {},
            "diners_club": {},
            "master_card": {},
            "union_pay": {}
          },
          "digital_wallet": {
            "accepts_nfc": {}
          }
        },
        "food_and_drink": {
          "alcohol": {
            "bar_service": {},
            "beer": {},
            "byo": {},
            "cocktails": {},
            "full_bar": {},
            "wine": {}
          },
          "meals": {
            "bar_snacks": {},
            "breakfast": {},
            "brunch": {},
            "lunch": {},
            "happy_hour": {},
            "dessert": {},
            "dinner": {},
            "tasting_menu": {}
          }
        },
        "services": {
          "delivery": {},
          "takeout": {},
          "drive_through": {},
          "dine_in": {
            "reservations": {},
            "online_reservations": {},
            "groups_only_reservations": {},
            "essential_reservations": {}
          }
        },
        "amenities": {
          "restroom": {},
          "smoking": {},
          "jukebox": {},
          "music": {},
          "live_music": {},
          "private_room": {},
          "outdoor_seating": {},
          "tvs": {},
          "atm": {},
          "coat_check": {},
          "wheelchair_accessible": {},
          "parking": {
            "parking": {},
            "street_parking": {},
            "valet_parking": {},
            "public_lot": {},
            "private_lot": {}
          },
          "sit_down_dining": {},
          "wifi": "string"
        },
        "attributes": {
          "business_meeting": "string",
          "clean": "string",
          "crowded": "string",
          "dates_popular": "string",
          "dressy": "string",
          "families_popular": "string",
          "gluten_free_diet": "string",
          "good_for_dogs": "string",
          "groups_popular": "string",
          "healthy_diet": "string",
          "late_night": "string",
          "noisy": "string",
          "quick_bite": "string",
          "romantic": "string",
          "service_quality": "string",
          "singles_popular": "string",
          "special_occasion": "string",
          "trendy": "string",
          "value_for_money": "string",
          "vegan_diet": "string",
          "vegetarian_diet": "string"
        }
      },
      "geocodes": {
        "drop_off": {
          "latitude": 0,
          "longitude": 0
        },
        "front_door": {
          "latitude": 0,
          "longitude": 0
        },
        "main": {
          "latitude": 0,
          "longitude": 0
        },
        "road": {
          "latitude": 0,
          "longitude": 0
        },
        "roof": {
          "latitude": 0,
          "longitude": 0
        }
      },
      "hours": {
        "display": "string",
        "is_local_holiday": true,
        "open_now": true,
        "regular": [
          {
            "close": "string",
            "day": 0,
            "open": "string"
          }
        ]
      },
      "hours_popular": [
        {
          "close": "string",
          "day": 0,
          "open": "string"
        }
      ],
      "link": "string",
      "location": {
        "address": "string",
        "address_extended": "string",
        "admin_region": "string",
        "census_block": "string",
        "country": "string",
        "cross_street": "string",
        "dma": "string",
        "formatted_address": "string",
        "locality": "string",
        "neighborhood": [
          "string"
        ],
        "po_box": "string",
        "post_town": "string",
        "postcode": "string",
        "region": "string"
      },
      "menu": "string",
      "name": "string",
      "photos": [
        {
          "id": "string",
          "created_at": "2023-10-11T16:32:04.439Z",
          "prefix": "string",
          "suffix": "string",
          "width": 0,
          "height": 0,
          "classifications": [
            "string"
          ],
          "tip": {
            "id": "string",
            "created_at": "2023-10-11T16:32:04.439Z",
            "text": "string",
            "url": "string",
            "lang": "string",
            "agree_count": 0,
            "disagree_count": 0
          }
        }
      ],
      "popularity": 0,
      "price": 0,
      "rating": 0,
      "related_places": {},
      "social_media": {
        "facebook_id": "string",
        "instagram": "string",
        "twitter": "string"
      },
      "stats": {
        "total_photos": 0,
        "total_ratings": 0,
        "total_tips": 0
      },
      "store_id": "string",
      "tastes": [
        "string"
      ],
      "tel": "string",
      "timezone": "string",
      "tips": [
        {
          "id": "string",
          "created_at": "2023-10-11T16:32:04.439Z",
          "text": "string",
          "url": "string",
          "lang": "string",
          "agree_count": 0,
          "disagree_count": 0
        }
      ],
      "venue_reality_bucket": "string", 
      "verified": true,
      "website": "string"
    }
  ],
  "context": {
    "geo_bounds": {
      "circle": {
        "center": {
          "latitude": 0,
          "longitude": 0
        },
        "radius": 0
      }
    }
  }
}


Migration Guide
The Places API is now built on the Open Source Places dataset, offering a variety of improvements to quality, available fields, and structure. API users can expect to see improved search results and greater accuracy for location-based queries. You can use the Places API in conjunction with the Users API to create an authenticated experience for you user.

Path Changes
The endpoints for the Places API and Users API have changed, notably removing the version segment (/v3/ or /v2/). Versioning will now be managed via the header, where you may specify a date. We will be releasing updated versions of the Place Match endpoint in the coming months.

Autocomplete Endpoints
Old Endpoint - Host: api.foursquare.com	New Endpoint- Host: places-api.foursquare.com
api.foursquare.com/v3/autocomplete	places-api.foursquare.com/autocomplete
Search & Data Endpoints
Old Endpoint - Host: api.foursquare.com	New Endpoint- Host: places-api.foursquare.com
api.foursquare.com/v3/places/search	places-api.foursquare.com/places/search
api.foursquare.com/v3/places/{fsq_id}	places-api.foursquare.com/places/{fsq_place_id}
api.foursquare.com/v3/places/{fsq_id}/tips	places-api.foursquare.com/places/{fsq_place_id}/tips
api.foursquare.com/v3/places/{fsq_id}/photos	places-api.foursquare.com/places/{fsq_place_id}/photos
Placemaker Endpoints (formerly Feedback Endpoints)
Old Endpoint - Host: api.foursquare.com	New Endpoint- Host: places-api.foursquare.com
api.foursquare.com/v3/places/{fsq_id}/proposeedit	places-api.foursquare.com/places/{fsq_place_id}/suggest/edit
api.foursquare.com/v3/places/{fsq_id}/flag	places-api.foursquare.com/places/{fsq_place_id}/suggest/remove
places-api.foursquare.com/places/{fsq_place_id}/suggest/merge
api.foursquare.com/v3/feedback/status	places-api.foursquare.com/suggest/status
--	places-api.foursquare.com/places/{fsq_place_id}/flag (NEW)
--	places-api.foursquare.com/places/suggest/place (NEW)
User Management Endpoints
Old Endpoint - Host: api.foursquare.com	New Endpoint- Host: users-api.foursquare.com
api.foursquare.com/v2/usermanagement/createuser	users-api.foursquare.com/users/managed-user/create
api.foursquare.com/v2/usermanagement/deleteuser	users-api.foursquare.com/users/managed-user/delete
api.foursquare.com/v2/usermanagement/refreshtoken	users-api.foursquare.com/users/managed-user/refresh-token
Geotagging Endpoints
Old Endpoint - Host: api.foursquare.com	New Endpoint- Host: places-api.foursquare.com
api.foursquare.com/v3/places/nearby	places-api.foursquare.com/geotagging/candidates
api.foursquare.com/v3/place/select	places-api.foursquare.com/geotagging/confirm
Auth Changes
We've migrated our authentication system from API keys to service keys, which offer enhanced security, better management, and more granular access control. The Legacy V3 Places API all support this new Service key Authentication method to make your migration easier.

Old Auth	New Auth
Method: API Keys	Method: Service Keys
Header: Authorization: <API_KEY>	Header: Authorization: Bearer <SERVICE_KEY>
Request Changes
Versioning
We’ve added date-based versioning to this API to ensure stability and compatibility; please include the version date in the header, e.g.:

Header Authenticatio String

X-Places-Api-Version: 2025-06-17
X-Users-Api-Version: 2025-06-17
Note that the versions are separated by API (Users vs. Places APIs) and may not always have the same version date.

We will update versions for breaking changes only. We will not update a version if a field gets added to a response.

Response Changes
The following changes have been made to the Response object

Fields
Old Field	New Field(s)	Notes
fsq_id	fsq_place_id	Consumers of our flat file schema will now find ID parity between the flat file and the API.
categories	categories (unchanged)	The ID returned in the categories JSON is now a BSON category ID instead of an integer. Categories are documented here.
location	location (unchanged)	The dma and census_block have been removed from this section of the response and will be moved to extended_location
photos	photos (unchanged)	We've released a new classification model on photos.
geocodes	latitude, longitude	These coordinates are the main entry point of the POI.
features	attributes	Renamed.
n/a	extended_location	New extended location object containing dma and census_block
timezone	Removed	No longer available.
closed_bucket	Removed	No longer available. Use the existence of date_closed instead.
fax	Removed	No longer available.
verified	Removed	No longer available.
Pricing Changes
You can learn more about the Places and Users APIs pricing here.

For customers who joined before June 3, 2025, you can expect the following changes on October 1, 2025:

V3 endpoints will remain at their current rate: $18.75 CPM.
V2 endpoints you currently pay for will be charged as Pro and Premium, following our new Places API pricing. See below for a list of Pro and Premium endpoints. Select V2 endpoints are free to use, including checkins and tips endpoints, and will remain at no cost.
For customers who joined between June 2, 2025, and June 17, 2025, your usage of V2 and V3 endpoints is charged at a $18.75 CPM. Any free V2 endpoints will remain at no cost.

For customers who joined after June 17, 2025, you do not have access to V3 endpoints and should not use the deprecated V2 endpoints. Any free V2 endpoints will remain at no cost.

V2 endpoints to be priced at Pro and Premium for customers who joined before June 3, 2025:
The following endpoints will be charged as Premium:

/v2/venues/X
/v2/venues/X/photos
/v2/venues/X/tips
/v2/venues/X/hours
/v2/venues/X/menu
/v2/venues/X/links
/v2/venues/X/events
/v2/venues/X/attributes
The following endpoints will be charged as Pro:

/v2/venues/search
/v2/search/recommendations
/v2/venues/trending
/v2/venues/X/related
Any checkins, lists, tastes, tips, and users endpoints will remain free.

If you have further questions about migration, please contact us for assistance.

open source code
duckdb
Prerequisites
Install DuckDB (version 1.4.0 or later)
Create a Places Portal Access Token
Click to open the page
Generate a new access token
Instructions
Step 1: Start DuckDB
In your terminal, start DuckDB by typing:
duckdb
Copy
You should see output similar to this, indicating a connection to an in-memory database:
DuckDB vX.Y.Z (build_hash)
Enter ".help" for usage hints.
Connected to a transient in-memory database.
Use ".open FILENAME" to reopen on a persistent database.
Copy
Step 2: Install and Load the httpfs Extension
The httpfs (HTTP File System) extension allows DuckDB to communicate over HTTP/HTTPS, which is necessary to connect to the Iceberg REST Catalog endpoint and potentially retrieve data from remote storage like S3.
Install and load the httpfs extension
INSTALL httpfs;
LOAD httpfs;
Copy
Step 3: Create an Iceberg Secret with Your Token
Replace YOUR_ACCESS_TOKEN with your Places Portal access token
CREATE SECRET iceberg_secret (
    TYPE ICEBERG,
    TOKEN "YOUR_ACCESS_TOKEN"
);
Copy
Step 4: Attach to the Iceberg Catalog
Now, you can attach your Iceberg REST Catalog to your DuckDB session. You will use the secret you just created for authentication.
ATTACH 'places' AS places (
    TYPE iceberg,
    SECRET iceberg_secret,
    ENDPOINT 'https://catalog.h3-hub.foursquare.com/iceberg'
);
Copy
Step 5: List All Tables (Optional)
You can now explore the tables within your attached Iceberg catalog
SELECT table_schema, table_name FROM information_schema.tables WHERE table_catalog = 'places';
Copy
This will show you a list of schemas and tables available in the Places Portal catalog.
Step 6: Query the OS Places Table
You can query this specific dataset
SELECT * FROM places.datasets.places_os LIMIT 1000;

pyspark
Prerequisites
Apache Spark 3.4 or later (The most recent versions of Iceberg work best with Spark 3.4+)
Iceberg Spark runtime JARs must be available:
Include iceberg-spark-runtime.jar matching your Spark version (e.g., iceberg-spark-runtime-3.4_2.12.jar) either:
via --packages on spark-submit or pyspark
or added to your cluster's libraries
PySpark installed (pip install pyspark)
Create a Places Portal Access Token
Click to open the page
Generate a new access token
Instructions
Step 1: Configure Spark Session
In your Python code, update the Spark configuration.
Replace YOUR_ACCESS_TOKEN with your Places Portal access token
token = "YOUR_ACCESS_TOKEN"

// Catalog definition
spark.conf.set("spark.sql.catalog.places", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set("spark.sql.catalog.places.type", "rest")
spark.conf.set("spark.sql.catalog.places.uri", "https://catalog.h3-hub.foursquare.com/iceberg")
spark.conf.set("spark.sql.catalog.places.token", token)
spark.conf.set("spark.sql.catalog.places.warehouse", "places")

// Enable S3 IO with vended creds
spark.conf.set("spark.sql.catalog.places.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
spark.conf.set("spark.sql.catalog.places.header.X-Iceberg-Access-Delegation", "true")

// Region (required for AWS SDK v2 bootstrap)
spark.conf.set("spark.sql.catalog.places.s3.region", "us-east-1")
spark.conf.set("spark.sql.catalog.places.client.region", "us-east-1") // extra safety for some versions  

# Set the default catalog for SQL queries 
spark.sql("USE places")
Copy
Step 2: List All Namespaces and Tables (Optional)
# List all namespaces
namespaces = spark.sql("SHOW NAMESPACES")
namespaces.show()

# Extract namespace names
namespaces_list = [row["namespace"] if "namespace" in row else row["databaseName"] for row in namespaces.collect()]

# List all tables in all namespaces
all_tables = []
for ns in namespaces_list:
    tables = spark.sql(f"SHOW TABLES IN {ns}")
    all_tables.extend(tables.collect())

# Create a DataFrame of all table metadata
all_tables_df = spark.createDataFrame(all_tables)
all_tables_df.show(truncate=False)
Copy
Step 3: Query the OS Places Table
# Read the table
df = spark.read.table("places.datasets.places_os")
df.limit(1000).show()
Copy

pyiceberg
Prerequisites
Python: Install Python (version 3.9 or newer)
Install Python dependencies
pip install pyiceberg pyarrow pandas
Optional: pip install polars
Create a Places Portal Access Token
Click to open the page
Generate a new access token
Instructions
Step 1: Create an Iceberg Secret with Your Token
In your Python code, create an Iceberg secret.
Replace YOUR_ACCESS_TOKEN with your Places Portal access token
from pyiceberg.catalog import load_catalog

catalog = load_catalog(
    "default",
    **{
        "warehouse": "places",
        "uri": "https://catalog.h3-hub.foursquare.com/iceberg",
        "token": "YOUR_ACCESS_TOKEN",
        "header.content-type": "application/vnd.api+json",
        "rest-metrics-reporting-enabled": "false",
    },
)
Copy
Step 2: List Namespaces and Tables (Optional)
namespaces = catalog.list_namespaces()

for namespace in namespaces:
    print(catalog.list_tables(namespace))
Copy
Prints a list of tuples for each namespace of the format <namespace>.<table>
Step 3: Query the OS Places Table
Using Pandas
Catalog tables can be accessed using the catalog.load_table() method and passing in a string of the format <namespace>.<table>:
table = catalog.load_table('datasets.places_os')
df = table.scan(limit=1000).to_pandas()
print(df)
Copy
Using Polars
You can use Polars for efficient, lazy evaluation and faster performance, especially on larger datasets.
import polars as pl 

# Use Polars lazy scanning from a PyIceberg table
table = catalog.load_table(('datasets', 'places_os'))
df = pl.scan_iceberg(table)
Copy
For more information on Polars lazy DataFrames and transformations, see the Polars User Guide
PyIceberg Tips and Best Practices
Start Small: Always use .limit() when exploring new datasets to avoid downloading large amounts of data.
Use Filters: Apply filters before calling to reduce data transfer and processing time. .to_pandas()
You can combine .filter() , .select() and .limit() fluently.