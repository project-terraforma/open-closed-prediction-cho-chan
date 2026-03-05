looking at overture feb release database

└─────────┴────────────────────┴──────────┘
┌────────────────────────┬──────────┬────────┬───────┬──────────┐
│         state          │   open   │ closed │ temp  │  total   │
│        varchar         │  int64   │ int64  │ int64 │  int64   │
├────────────────────────┼──────────┼────────┼───────┼──────────┤
│ NULL                   │ 34137431 │      0 │     0 │ 34137431 │
│ CA                     │  1616075 │     14 │     0 │  1616089 │
│ SP                     │  1529729 │      0 │     0 │  1529729 │
│ ENG                    │  1476075 │      0 │     0 │  1476075 │
│ TX                     │  1381775 │     78 │     0 │  1381854 │
│ FL                     │  1129677 │     67 │     0 │  1129744 │
│                        │  1052869 │      0 │     0 │  1052869 │
│ NY                     │   840233 │      8 │     0 │   840241 │
│ PA                     │   699384 │      6 │     0 │   699390 │
│ ON                     │   601095 │    106 │     0 │   601203 │
│ IL                     │   588006 │      6 │     0 │   588012 │
│ SC                     │   570314 │      3 │     0 │   570317 │
│ OH                     │   529508 │     32 │     0 │   529541 │
│ NC                     │   518834 │     14 │     0 │   518848 │
│ GA                     │   506672 │     20 │     0 │   506692 │
│ MH                     │   502988 │      0 │     0 │   502988 │
│ MG                     │   483689 │      0 │     0 │   483689 │
│ MI                     │   474215 │      3 │     0 │   474218 │
│ WA                     │   455008 │     15 │     0 │   455023 │
│ RS                     │   432514 │      0 │     0 │   432514 │
│ ·                      │        · │      · │     · │        · │
│ ·                      │        · │      · │     · │        · │
│ ·                      │        · │      · │     · │        · │

─────────┬──────────────┬────────────────┬────────────────┬────────────────┐
│ region  │ closed_count │ avg_confidence │ min_confidence │ max_confidence │
│ varchar │    int64     │     double     │     double     │     double     │
├─────────┼──────────────┼────────────────┼────────────────┼────────────────┤
│ ON      │          106 │         0.9514 │            0.9 │         0.9981 │
│ MO      │           93 │          0.913 │         0.0909 │         0.9947 │
│ TX      │           78 │         0.9517 │         0.1918 │         0.9981 │
│ FL      │           67 │         0.9557 │            0.9 │         0.9968 │
│ AB      │           59 │         0.9573 │            0.9 │         0.9981 │
│ QC      │           58 │          0.948 │            0.9 │         0.9981 │
│ OH      │           32 │         0.9543 │            0.9 │         0.9947 │
│ BC      │           27 │         0.9495 │            0.9 │         0.9947 │
│ GA      │           20 │         0.9649 │            0.9 │         0.9947 │
│ WA      │           15 │         0.8482 │         0.1094 │          0.977 │
│ VA      │           15 │         0.9253 │         0.2791 │         0.9981 │
│ CA      │           14 │         0.9686 │            0.9 │         0.9947 │
│ NC      │           14 │          0.919 │         0.3636 │         0.9947 │
│ AL      │           12 │         0.9576 │            0.9 │         0.9947 │
│ MS      │           10 │         0.9745 │            0.9 │         0.9947 │
│ NS      │            9 │         0.9464 │            0.9 │         0.9947 │
│ NY      │            8 │         0.9503 │            0.9 │         0.9947 │
│ MN      │            8 │         0.9578 │            0.9 │          0.977 │
│ LA      │            8 │         0.8992 │         0.3744 │         0.9947 │
│ NJ      │            7 │         0.9355 │            0.9 │         0.9947 │
│ ·       │            · │             ·  │             ·  │             ·  │
│ ·       │            · │             ·  │             ·  │             ·  │
│ ·       │            · │             ·  │             ·  │             ·  │
│ QLD     │            3 │            0.9 │            0.9 │            0.9 │
│ SC      │            3 │         0.9888 │          0.977 │         0.9947 │
│ NSW     │            3 │         0.9631 │            0.9 │         0.9947 │
│ MI      │            3 │         0.9513 │            0.9 │          0.977 │
│ BOP     │            2 │         0.9385 │            0.9 │          0.977 │
│ MA      │            2 │         0.9859 │          0.977 │         0.9947 │
│ ID      │            2 │         0.9474 │            0.9 │         0.9947 │
│ NE      │            2 │         0.9848 │         0.9749 │         0.9947 │
│ DE      │            2 │          0.977 │          0.977 │          0.977 │
│ UT      │            1 │         0.9947 │         0.9947 │         0.9947 │
│ NM      │            1 │          0.977 │          0.977 │          0.977 │
│ YT      │            1 │          0.977 │          0.977 │          0.977 │
│ SA      │            1 │            0.9 │            0.9 │            0.9 │
│ IA      │            1 │            0.9 │            0.9 │            0.9 │
│ NV      │            1 │          0.977 │          0.977 │          0.977 │
│ VT      │            1 │          0.977 │          0.977 │          0.977 │
│ NH      │            1 │            0.9 │            0.9 │            0.9 │
│ PR      │            1 │            0.9 │            0.9 │            0.9 │
│ CAN     │            1 │            0.9 │            0.9 │            0.9 │
│ AZ      │            1 │          0.977 │          0.977 │          0.977 │
├─────────┴──────────────┴────────────────┴────────────────┴────────────────┤
│ 56 rows (40 shown)                                              5 columns │
└───────────────────────────────────────────────────────────────────────────┘

File: ../data/overture-feb-release-closed.json
Total: 785  |  open: 0  |  closed: 785

region        count     pct
---------------------------
ON              106   13.5%
MO               93   11.8%
TX               78    9.9%
FL               67    8.5%
AB               59    7.5%
QC               58    7.4%
OH               32    4.1%
BC               27    3.4%
GA               20    2.5%
VA               15    1.9%
WA               15    1.9%
CA               14    1.8%
NC               14    1.8%
AL               12    1.5%
MS               10    1.3%
NS                9    1.1%
LA                8    1.0%
MN                8    1.0%
NY                8    1.0%
OK                7    0.9%
TN                7    0.9%
KS                7    0.9%
NJ                7    0.9%
MB                7    0.9%
AR                6    0.8%
SK                6    0.8%
PA                6    0.8%
IN                6    0.8%
MD                6    0.8%
IL                6    0.8%
KY                6    0.8%
NB                5    0.6%
NL                5    0.6%
Singapore         5    0.6%
AUK               4    0.5%
PE                3    0.4%
MI                3    0.4%
SC                3    0.4%
NSW               3    0.4%
QLD               3    0.4%
NE                2    0.3%
ID                2    0.3%
MA                2    0.3%
DE                2    0.3%
BOP               2    0.3%
NM                1    0.1%
AZ                1    0.1%
IA                1    0.1%
UT                1    0.1%
NV                1    0.1%
YT                1    0.1%
VT                1    0.1%
NH                1    0.1%
PR                1    0.1%
CAN               1    0.1%
SA                1    0.1%

category                                 closed_count
------------------------------------------------------
shoe_store                                        141
insurance_agency                                   96
home_improvement_store                             71
public_health_clinic                               69
grocery_store                                      66
baby_gear_and_furniture                            52
toy_store                                          36
home_goods_store                                   26
banks                                              24
health_food_restaurant                             21
department_store                                   18
shipping_collection_services                       16
courier_and_delivery_services                      15
mexican_restaurant                                 12
assisted_living_facility                           12
medical_center                                     10
mortgage_lender                                     8
desserts                                            7
shipping_center                                     6
urgent_care_clinic                                  6
gym                                                 6
atms                                                5
real_estate_agent                                   5
pizza_delivery_service                              5
doctor                                              5
restaurant                                          4
orthopedist                                         4
chicken_wings_restaurant                            3
liquor_store                                        3
kitchen_supply_store                                2
driving_school                                      2
mattress_store                                      2
rehabilitation_center                               2
financial_advising                                  2
chiropractor                                        2
pharmacy                                            2
pediatrician                                        2
personal_injury_law                                 1
tobacco_shop                                        1
swimming_pool                                       1
supermarket                                         1
hearing_aids                                        1
neurologist                                         1
business_consulting                                 1
home_developer                                      1
employment_agencies                                 1
credit_union                                        1
childrens_hospital                                  1
endoscopist                                         1
diagnostic_imaging                                  1
surgical_center                                     1
gunsmith                                            1
sports_medicine                                     1
outlet_store                                        1

┌────────────────────────────┬─────────────┐
│           state            │ temp_closed │
│          varchar           │    int64    │
├────────────────────────────┼─────────────┤
│ QC                         │           7 │
│ ON                         │           2 │
│ AB                         │           2 │
│ OH                         │           1 │
│ TX                         │           1 │
│ BC                         │           1 │
│ ON                         │           1 │
│ SK                         │           1 │
│ NS                         │           1 │
│ CO                         │           1 │
│ OK                         │           1 │
│ KY                         │           1 │
│ METROPOLITANA              │           0 │
│ Edo de México              │           0 │
│ Región metropolitana       │           0 │
│ Quinta Región de Val       │           0 │
│ SP                         │           0 │
│ Región De Valparaiso       │           0 │
│ Región de Valparaiso       │           0 │
│ Metropolitan Region        │           0 │
│    ·                       │           · │
│    ·                       │           · │
│    ·                       │           · │
│ Schonen                    │           0 │