import os
import pandas as pd


def collect_counties(state, app):
    p_counties = os.path.join(
        "/projects/dwind/data/counties_by_state", 
        f"{state}.csv"
    )
    df = pd.read_csv(p_counties, dtype=str)
    
    p_agents = '/projects/dwind/agents/'
    agents = []
    
    for _, row in df.iterrows():
        county = row['coutnty_file']
        fips = str(row['fips_code'])
        
        states = [
            'alabama', 
            'alaska', 
            'arizona', 
            'arkansas', 
            'california', 
            'colorado', 
            'connecticut'
        ]
        
        if state in states:
            fips = "0" + fips
            
        p = os.path.join(p_agents, state + '_' + county)
        if os.path.exists(p):
            p = os.path.join(p, f'agents_dwind_{app}.parquet')
            
            if os.path.exists(p):
                agent_df = pd.read_parquet(p)
                agent_df['state'] = state
                agent_df['county_file'] = county
                agent_df['fips_code'] = fips
                
                # fix county_id
                county_f = '/projects/dwind/configs/costs/dgen_cnty_fips_mapping.csv'
                county_df = pd.read_csv(county_f, dtype={'fips_code': str})
                
                county_df = county_df[['fips_code', 'county_id']]
                county_df['fips_code'] = county_df['fips_code'].astype(str)

                agent_df = agent_df.drop(columns=['county_id'])
                agent_df['fips_code'] = agent_df['fips_code'].astype(str)
                agent_df = agent_df.merge(
                    county_df, 
                    on='fips_code', 
                    how='left'
                )
                
                # fix rev_index
                rev_f = '/projects/dwind/configs/rev/wind/lkup_rev_index_to_gid.csv'
                rev_df = pd.read_csv(rev_f)
                agent_df = agent_df.merge(
                    rev_df, 
                    on='rev_gid_wind',
                    how='left'
                )
                
                # add BA
                ba_f = '/projects/dwind/configs/costs/county_to_ba_mapping.csv'
                ba_df = pd.read_csv(ba_f)
                agent_df = agent_df.merge(
                    ba_df, 
                    on='county_id', 
                    how='left'
                )
                
                # remove agents without rate ids
                if app == 'btm':
                    agent_df = agent_df[~agent_df['rate_id_alias'].isna()]
                
                agents.append(agent_df)
                
    agents_df = pd.concat(agents)
    
    return agents_df


def collect_counties_by_state(state):
    p_agents = '/projects/dwind/agents/'
    p_state = os.path.join(p_agents, state)
    if not os.path.exists(p_state):
        os.mkdir(p_state)
    
    # save fom agents by state
    p_state_fom = os.path.join(p_state, 'agents_dwind_fom.pickle')
    # if not os.path.exists(p_state_fom):
    agents_fom_df = collect_counties(state, 'fom')
    agents_fom_df.to_pickle(p_state_fom, protocol=3)
    
    agents_fom_df = pd.read_pickle(p_state_fom)
    print(state, 'fom', len(agents_fom_df.index))
    
    # save btm agents by state
    p_state_btm = os.path.join(p_state, 'agents_dwind_btm.pickle')
    # if not os.path.exists(p_state_btm):
    agents_btm_df = collect_counties(state, 'btm')
    agents_btm_df.to_pickle(p_state_btm, protocol=3)
    
    agents_btm_df = pd.read_pickle(p_state_btm)
    print(state, 'btm', len(agents_btm_df.index))
    
                
states = [    
    'minnesota',
    'wisconsin',
    'iowa',
    'illinois',
    'indiana',
    'michigan',
    'ohio',
    'pennsylvania',
    'new_york',
    'kentucky',
    'california',
    'north_dakota',
    'south_dakota',
    'nebraska',
    'kansas',
    'oklahoma',
    'texas',
    'arkansas',
    'missouri',
    'montana',
    'wyoming',
    'colorado',
    'new_mexico',
    'idaho',
    'washington',
    'oregon',
    'maine',
    'new_hampshire',
    'vermont',
    'massachusetts',
    'connecticut',
    'rhode_island',
    'arizona',
    'nevada',
    'utah',
    'new_jersey',
    'delaware',
    'maryland',
    'virginia',
    'west_virginia',
    'districtofcolumbia',
    'tennessee',
    'north_carolina',
    'south_carolina',
    'georgia'
    'florida',
    'alabama',
    'mississippi',
    'louisiana'
]

states = ['new_mexico']
for state in states:
    collect_counties_by_state(state)

'''
minnesota fom 940082
minnesota btm 439394
minnesota fom 505985 (old)
minnesota btm 195205 (old)
wisconsin fom 975059
wisconsin btm 492396
wisconsin fom 477407 (old)
wisconsin btm 192474 (old)
iowa fom 945903
iowa btm 263702
iowa fom 711339 (old)
iowa btm 162921 (old)
illinois fom 838447
illinois btm 160242
illinois fom 512308 (old)
illinois btm 240236 (old)
indiana fom 745402
indiana btm 337940
indiana fom 359490 (old)
indiana btm 138564 (old)
michigan fom 865763
michigan btm 539074
michigan fom 333964 (old)
michigan btm 182186 (old)
ohio fom 800255
ohio btm 472574
ohio fom 332783 (old)
ohio btm 178528 (old)
pennsylvania fom 489996
pennsylvania btm 300626
pennsylvania fom 146599 (old)
pennsylvania btm 93707 (old)
new_york fom 564569
new_york btm 366952
new_york fom 144742 (old)
new_york btm 87812 (old)
kentucky fom 303903
kentucky btm 220254
kentucky fom 109020 (old)
kentucky btm 84876 (old)
california fom 1010219
california btm 478479
california fom 333800 (old)
california btm 134959 (old)

north_dakota fom 320518
north_dakota btm 69211
north_dakota fom 244788 (old)
north_dakota btm 95727 (old)
south_dakota fom 269254
south_dakota btm 47579
south_dakota fom 207111 (old)
south_dakota btm 45689 (old)
nebraska fom 383986
nebraska btm 70382
nebraska fom 304144 (old)
nebraska btm 75135 (old)
kansas fom 420146
kansas btm 164407
kansas fom 261194 (old)
kansas btm 113877 (old)
oklahoma fom 661317
oklahoma btm 314325
oklahoma fom 304411 (old)
oklahoma btm 122653 (old)
texas fom 1935369
texas btm 971774
texas fom 642399 (old)
texas btm 279135 (old)
arkansas fom 547142
arkansas btm 217803
arkansas fom 228183 (old)
arkansas btm 73946 (old)
missouri fom 779916
missouri btm 328505
missouri fom 355264 (old)
missouri btm 176228 (old)

montana fom 333101
montana btm 55976
montana fom 212335 (old)
montana btm 44402 (old)
wyoming fom 103115
wyoming btm 25101
wyoming fom 37830 (old)
wyoming btm 22168 (old)
colorado fom 377493
colorado btm 196745
colorado fom 144831 (old)
colorado btm 63504 (old)
new_mexico fom 331754
new_mexico btm 71439
new_mexico fom 92802 (old)
new_mexico btm 35476 (old)
idaho fom 234445
idaho btm 97534
idaho fom 79692 (old)
idaho btm 46354 (old)

washington fom 392807
washington btm 231922
washington fom 93633 (old)
washington btm 47024 (old)
oregon fom 219954
oregon btm 129054
oregon fom 68310 (old)
oregon btm 43555 (old)

maine fom 118916
maine btm 87398
maine fom 27695 (old)
maine btm 16073 (old)
new_hampshire fom 73271
new_hampshire btm 64085
new_hampshire fom 9740 (old)
new_hampshire btm 8183 (old)
vermont fom 50847
vermont btm 46654
vermont fom 14320 (old)
vermont btm 12818 (old)
massachusetts fom 122565
massachusetts btm 82852
massachusetts fom 22908 (old)
massachusetts btm 12219 (old)
connecticut fom 82895
connecticut btm 62940
connecticut fom 12741 (old)
connecticut btm 8817 (old)
rhode_island fom 18999
rhode_island btm 13568
rhode_island fom 4276 (old)
rhode_island btm 2742 (old)

arizona fom 538775
arizona btm 198795
arizona fom 114412 (old)
arizona btm 44556 (old)
nevada fom 139282
nevada btm 55688
nevada fom 54859 (old)
nevada btm 10444 (old)
utah fom 187738
utah btm 32418
utah fom 63820 (old)
utah btm 4011 (old)

new_jersey fom 124538
new_jersey btm 50816
new_jersey fom 24369 (old)
new_jersey btm 8514 (old)
delaware fom 36782
delaware btm 28358
delaware fom 8967 (old)
delaware btm 7192 (old)
maryland fom 202849
maryland btm 136697
maryland fom 52550 (old)
maryland btm 30900 (old)
virginia fom 515879
virginia btm 292862
virginia fom 71539 (old)
virginia btm 29357 (old)
west_virginia fom 78113
west_virginia btm 43081
west_virginia fom 10109 (old)
west_virginia btm 5432 (old)
districtofcolumbia fom 1825
districtofcolumbia btm 1419
districtofcolumbia fom 241 (old)
districtofcolumbia btm 170 (old)

tennessee fom 485632
tennessee btm 296271
tennessee fom 79832 (old)
tennessee btm 45384 (old)
north_carolina fom 708681
north_carolina btm 256774
north_carolina fom 70342 (old)
north_carolina btm 23619 (old)
south_carolina fom 402376
south_carolina btm 152156
south_carolina fom 66263 (old)
south_carolina btm 22105 (old)
georgia fom 556078
georgia btm 333820
georgia fom 91293 (old)
georgia btm 49379 (old)
florida fom 624394
florida btm 311406
florida fom 74910 (old)
florida btm 35835 (old)
alabama fom 622177
alabama btm 188023
alabama fom 177548 (old)
alabama btm 39445 (old)
mississippi fom 564047
mississippi btm 169841
mississippi fom 76695 (old)
mississippi btm 18006 (old)
louisiana fom 244131
louisiana btm 129898
louisiana fom 26400 (old)
louisiana btm 13720 (old)
'''