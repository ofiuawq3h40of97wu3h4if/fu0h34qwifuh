bert_version = 'bert-base-cased'
framenet_path = "./data/fndata-1.7"
cache_dir = "./cache/"

OPENSESAME_TEST_FILES = ["ANC__110CYL067.xml","ANC__110CYL069.xml","ANC__112C-L013.xml",
                            "ANC__IntroHongKong.xml","ANC__StephanopoulosCrimes.xml","ANC__WhereToHongKong.xml",
                            "KBEval__atm.xml","KBEval__Brandeis.xml","KBEval__cycorp.xml","KBEval__parc.xml",
                            "KBEval__Stanford.xml","KBEval__utd-icsi.xml","LUCorpus-v0.3__20000410_nyt-NEW.xml",
                            "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml","LUCorpus-v0.3__enron-thread-159550.xml",
                            "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml","LUCorpus-v0.3__SNO-525.xml",
                            "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml","Miscellaneous__Hound-Ch14.xml",
                            "Miscellaneous__SadatAssassination.xml","NTI__NorthKorea_Introduction.xml",
                            "NTI__Syria_NuclearOverview.xml","PropBank__AetnaLifeAndCasualty.xml"]

OPENSESAME_DEV_FILES = ["ANC__110CYL072.xml","KBEval__MIT.xml","LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
                            "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml","Miscellaneous__Hijack.xml",
                            "NTI__NorthKorea_NuclearOverview.xml","NTI__WMDNews_062606.xml",
                            "PropBank__TicketSplitting.xml"]

OPENSESAME_TRAIN_FILES = ['ANC__110CYL068.xml', 'ANC__110CYL070.xml', 'ANC__110CYL200.xml', 'ANC__112C-L012.xml', 
                            'ANC__chapter1_911report.xml', 'ANC__chapter8_911report.xml', 'ANC__EntrepreneurAsMadonna.xml', 
                            'ANC__HistoryOfGreece.xml', 'ANC__HistoryOfJerusalem.xml', 'ANC__HistoryOfLasVegas.xml', 
                            'ANC__IntroJamaica.xml', 'ANC__IntroOfDublin.xml', 'ANC__journal_christine.xml', 
                            'ANC__WhatToHongKong.xml', 'fullText.xsl', 'KBEval__LCC-M.xml', 'KBEval__lcch.xml', 
                            'LUCorpus-v0.3__20000416_xin_eng-NEW.xml', 'LUCorpus-v0.3__20000419_apw_eng-NEW.xml', 
                            'LUCorpus-v0.3__20000420_xin_eng-NEW.xml', 'LUCorpus-v0.3__20000424_nyt-NEW.xml', 
                            'LUCorpus-v0.3__602CZL285-1.xml', 'LUCorpus-v0.3__AFGP-2002-600002-Trans.xml', 
                            'LUCorpus-v0.3__AFGP-2002-600045-Trans.xml', 'LUCorpus-v0.3__AFGP-2002-600175-Trans.xml', 
                            'LUCorpus-v0.3__artb_004_A1_E1_NEW.xml', 'LUCorpus-v0.3__artb_004_A1_E2_NEW.xml', 
                            'LUCorpus-v0.3__CNN_AARONBROWN_ENG_20051101_215800.partial-NEW.xml', 
                            'LUCorpus-v0.3__CNN_ENG_20030614_173123.4-NEW-1.xml', 'LUCorpus-v0.3__wsj_1640.mrg-NEW.xml', 
                            'LUCorpus-v0.3__wsj_2465.xml', 'Miscellaneous__C-4Text.xml', 'Miscellaneous__Examples4-5.xml', 
                            'Miscellaneous__IranRelatedQuestions.xml', 'Miscellaneous__Orwell_1984_p1.xml', 
                            'Miscellaneous__Pickett.xml', 'Miscellaneous__SemAnno_1.xml', 
                            'Miscellaneous__Tiger_Of_San_Pedro.xml', 'Miscellaneous__tradeBalance020417.xml', 
                            'NTI__BWTutorial_chapter1.xml', 'NTI__ChinaOverview.xml', 'NTI__Iran_Biological.xml', 
                            'NTI__Iran_Chemical.xml', 'NTI__Iran_Introduction.xml', 'NTI__Iran_Missile.xml', 
                            'NTI__Iran_Nuclear.xml', 'NTI__Kazakhstan.xml', 'NTI__LibyaCountry1.xml', 
                            'NTI__NorthKorea_ChemicalOverview.xml', 'NTI__NorthKorea_NuclearCapabilities.xml', 
                            'NTI__Russia_Introduction.xml', 'NTI__SouthAfrica_Introduction.xml', 'NTI__Taiwan_Introduction.xml', 
                            'NTI__WMDNews_042106.xml', 'NTI__workAdvances.xml', 'PropBank__BellRinging.xml', 
                            'PropBank__ElectionVictory.xml', 'PropBank__LomaPrieta.xml', 'PropBank__PolemicProgressiveEducation.xml', 
                            'WikiTexts__acquisition.n.xml', 'WikiTexts__boutique.n.xml', 'WikiTexts__extent.n.xml', 
                            'WikiTexts__Fires_1.xml', 'WikiTexts__Fires_10.xml', 'WikiTexts__Fires_2.xml', 
                            'WikiTexts__Fires_3.xml', 'WikiTexts__Fires_4.xml', 'WikiTexts__Fires_5.xml', 'WikiTexts__Fires_6.xml', 
                            'WikiTexts__Fires_7.xml', 'WikiTexts__Fires_8.xml', 'WikiTexts__Fires_9.xml', 'WikiTexts__fund.n.xml', 
                            'WikiTexts__invoice.n.xml', 'WikiTexts__oven.n.xml', 'WikiTexts__someone.n.xml', 'WikiTexts__spatula.n.xml']

out_of_domain = ['Process_resume', 'Abusing', 'Within_distance', 'Forgiveness', 
                     'Board_vehicle', 'Proper_reference', 'Rape', 'Giving_birth', 
                     'Expected_location_of_person', 'Beyond_compare', 
                     'Render_nonfunctional', 'Trying_out', 'Intercepting', 'Robbery', 
                     'Waking_up', 'Fastener', 'Renting', 'Assemble', 'Convey_importance', 
                     'Exchange_currency', 'Disembarking', 'Sound_level', 'Adjusting', 
                     'Addiction', 'Growing_food', 'Forging', 'Out_of_existence', 
                     'Manner_of_life', 'Going_back_on_a_commitment', 'Capacity', 
                     'Change_direction', 'Being_in_category', 'Recovery', 
                     'People_by_morality', 'Reparation', 'Create_representation', 
                     'Emanating', 'Immobilization', 'Body_description_holistic', 
                     'Tasting', 'Rotting']

# US english to British english
us2gb = {'accessorize': 'accessorise', 'accessorized': 'accessorised', 'accessorizes': 
    'accessorises', 'accessorizing': 'accessorising', 'acclimatization': 'acclimatisation', 
    'acclimatize': 'acclimatise', 'acclimatized': 'acclimatised', 'acclimatizes': 'acclimatises', 
    'acclimatizing': 'acclimatising', 'accouterments': 'accoutrements', 'aerogram': 'aerogramme', 
    'aerograms': 'aerogrammes', 'aggrandizement': 'aggrandisement', 'aging': 'ageing', 'agonize': 
    'agonise', 'agonized': 'agonised', 'agonizes': 'agonises', 'agonizing': 'agonising', 'agonizingly': 
    'agonisingly', 'airplane': 'aeroplane', 'airplanes ': 'aeroplanes ', 'almanac': 'almanack', 
    'almanacs': 'almanacks', 'aluminum': 'aluminium', 'amortizable': 'amortisable', 'amortization': 
    'amortisation', 'amortizations': 'amortisations', 'amortize': 'amortise', 'amortized': 'amortised', 
    'amortizes': 'amortises', 'amortizing': 'amortising', 'amphitheater': 'amphitheatre', 
    'amphitheaters': 'amphitheatres', 'analog': 'analogue', 'analogs': 'analogues', 'analyze': 
    'analyse', 'analyzed': 'analysed', 'analyzes': 'analyses', 'analyzing': 'analysing', 'anemia': 
    'anaemia', 'anemic': 'anaemic', 'anesthesia': 'anaesthesia', 'anesthetic': 'anaesthetic', 
    'anesthetics': 'anaesthetics', 'anesthetist': 'anaesthetist', 'anesthetists': 'anaesthetists', 
    'anesthetize': 'anaesthetize', 'anesthetized': 'anaesthetized', 'anesthetizes': 'anaesthetizes', 
    'anesthetizing': 'anaesthetizing', 'anglicize': 'anglicise', 'anglicized': 'anglicised', 
    'anglicizes': 'anglicises', 'anglicizing': 'anglicising', 'annualized': 'annualised', 'antagonize': 
    'antagonise', 'antagonized': 'antagonised', 'antagonizes': 'antagonises', 'antagonizing': 
    'antagonising', 'apologize': 'apologise', 'apologized': 'apologised', 'apologizes': 'apologises', 
    'apologizing': 'apologising', 'appall': 'appal', 'appalls': 'appals', 'appetizer': 'appetiser', 
    'appetizers': 'appetisers', 'appetizing': 'appetising', 'appetizingly': 'appetisingly', 'arbor': 
    'arbour', 'arbors': 'arbours', 'archeological': 'archaeological', 'archeologically': 
    'archaeologically', 'archeologist': 'archaeologist', 'archeologists': 'archaeologists', 
    'archeology': 'archaeology', 'ardor': 'ardour', 'armor': 'armour', 'armored': 'armoured', 'armorer': 
    'armourer', 'armorers': 'armourers', 'armories': 'armouries', 'armory': 'armoury', 'artifact': 
    'artefact', 'artifacts': 'artefacts', 'authorize': 'authorise', 'authorized': 'authorised', 
    'authorizes': 'authorises', 'authorizing': 'authorising', 'ax': 'axe', 'backpedaled': 
    'backpedalled', 'backpedaling': 'backpedalling', 'balk': 'baulk', 'balked': 'baulked', 'balking': 
    'baulking', 'balks': 'baulks', 'banister': 'bannister', 'banisters': 'bannisters', 'baptize': 
    'baptise', 'baptized': 'baptised', 'baptizes': 'baptises', 'baptizing': 'baptising', 'bastardize': 
    'bastardise', 'bastardized': 'bastardised', 'bastardizes': 'bastardises', 'bastardizing': 
    'bastardising', 'battleax': 'battleaxe', 'bedeviled': 'bedevilled', 'bedeviling': 'bedevilling', 
    'behavior': 'behaviour', 'behavioral': 'behavioural', 'behaviorism': 'behaviourism', 'behaviorist': 
    'behaviourist', 'behaviorists': 'behaviourists', 'behaviors': 'behaviours', 'behoove': 'behove', 
    'behooved': 'behoved', 'behooves': 'behoves', 'bejeweled': 'bejewelled', 'belabor': 'belabour', 
    'belabored': 'belaboured', 'belaboring': 'belabouring', 'belabors': 'belabours', 'beveled': 
    'bevelled', 'bevies': 'bevvies', 'bevy': 'bevvy', 'biased': 'biassed', 'biasing': 'biassing', 
    'binging': 'bingeing', 'bougainvillea': 'bougainvillaea', 'bougainvilleas': 'bougainvillaeas', 
    'bowdlerize': 'bowdlerise', 'bowdlerized': 'bowdlerised', 'bowdlerizes': 'bowdlerises', 
    'bowdlerizing': 'bowdlerising', 'breathalyze': 'breathalyse', 'breathalyzed': 'breathalysed', 
    'breathalyzer': 'breathalyser', 'breathalyzers': 'breathalysers', 'breathalyzes': 'breathalyses', 
    'breathalyzing': 'breathalysing', 'brutalize': 'brutalise', 'brutalized': 'brutalised', 
    'brutalizes': 'brutalises', 'brutalizing': 'brutalising', 'busses': 'buses', 'bussing': 'busing', 
    'caliber': 'calibre', 'calibers': 'calibres', 'caliper': 'calliper', 'calipers': 'callipers', 
    'calisthenics': 'callisthenics', 'canalize': 'canalise', 'canalized': 'canalised', 'canalizes': 
    'canalises', 'canalizing': 'canalising', 'cancelation': 'cancellation', 'cancelations': 
    'cancellations', 'canceled': 'cancelled', 'canceling': 'cancelling', 'candor': 'candour', 
    'cannibalize': 'cannibalise', 'cannibalized': 'cannibalised', 'cannibalizes': 'cannibalises', 
    'cannibalizing': 'cannibalising', 'canonize': 'canonise', 'canonized': 'canonised', 'canonizes': 
    'canonises', 'canonizing': 'canonising', 'capitalize': 'capitalise', 'capitalized': 'capitalised', 
    'capitalizes': 'capitalises', 'capitalizing': 'capitalising', 'caramelize': 'caramelise', 
    'caramelized': 'caramelised', 'caramelizes': 'caramelises', 'caramelizing': 'caramelising', 
    'carbonize': 'carbonise', 'carbonized': 'carbonised', 'carbonizes': 'carbonises', 'carbonizing': 
    'carbonising', 'caroled': 'carolled', 'caroling': 'carolling', 'catalog': 'catalogue', 'cataloged': 
    'catalogued', 'cataloging': 'cataloguing', 'catalogs': 'catalogues', 'catalyze': 'catalyse', 
    'catalyzed': 'catalysed', 'catalyzes': 'catalyses', 'catalyzing': 'catalysing', 'categorize': 
    'categorise', 'categorized': 'categorised', 'categorizes': 'categorises', 'categorizing': 
    'categorising', 'cauterize': 'cauterise', 'cauterized': 'cauterised', 'cauterizes': 'cauterises', 
    'cauterizing': 'cauterising', 'caviled': 'cavilled', 'caviling': 'cavilling', 'center': 'centre', 
    'centered': 'centred', 'centerfold': 'centrefold', 'centerfolds': 'centrefolds', 'centerpiece': 
    'centrepiece', 'centerpieces': 'centrepieces', 'centers': 'centres', 'centigram': 'centigramme', 
    'centigrams': 'centigrammes', 'centiliter': 'centilitre', 'centiliters': 'centilitres', 
    'centimeter': 'centimetre', 'centimeters': 'centimetres', 'centralize': 'centralise', 'centralized': 
    'centralised', 'centralizes': 'centralises', 'centralizing': 'centralising', 'cesarean': 
    'caesarean', 'cesareans': 'caesareans', 'channeled': 'channelled', 'channeling': 'channelling', 
    'characterize': 'characterise', 'characterized': 'characterised', 'characterizes': 'characterises', 
    'characterizing': 'characterising', 'check': 'cheque', 'checkbook': 'chequebook', 'checkbooks': 
    'chequebooks', 'checkered': 'chequered', 'checks': 'cheques', 'chili': 'chilli', 'chimera': 
    'chimaera', 'chimeras': 'chimaeras', 'chiseled': 'chiselled', 'chiseling': 'chiselling', 'cipher': 
    'cypher', 'ciphers': 'cyphers', 'circularize': 'circularise', 'circularized': 'circularised', 
    'circularizes': 'circularises', 'circularizing': 'circularising', 'civilize': 'civilise', 
    'civilized': 'civilised', 'civilizes': 'civilises', 'civilizing': 'civilising', 'clamor': 'clamour', 
    'clamored': 'clamoured', 'clamoring': 'clamouring', 'clamors': 'clamours', 'clangor': 'clangour', 
    'clarinetist': 'clarinettist', 'clarinetists': 'clarinettists', 'collectivize': 'collectivise', 
    'collectivized': 'collectivised', 'collectivizes': 'collectivises', 'collectivizing': 
    'collectivising', 'colonization': 'colonisation', 'colonize': 'colonise', 'colonized': 'colonised', 
    'colonizer': 'coloniser', 'colonizers': 'colonisers', 'colonizes': 'colonises', 'colonizing': 
    'colonising', 'color': 'colour', 'colorant': 'colourant', 'colorants': 'colourants', 'colored': 
    'coloured', 'coloreds': 'coloureds', 'colorful': 'colourful', 'colorfully': 'colourfully', 
    'coloring': 'colouring', 'colorize': 'colourize', 'colorized': 'colourized', 'colorizes': 
    'colourizes', 'colorizing': 'colourizing', 'colorless': 'colourless', 'colors': 'colours', 
    'commercialize': 'commercialise', 'commercialized': 'commercialised', 'commercializes': 
    'commercialises', 'commercializing': 'commercialising', 'compartmentalize': 'compartmentalise', 
    'compartmentalized': 'compartmentalised', 'compartmentalizes': 'compartmentalises', 
    'compartmentalizing': 'compartmentalising', 'computerize': 'computerise', 'computerized': 
    'computerised', 'computerizes': 'computerises', 'computerizing': 'computerising', 'conceptualize': 
    'conceptualise', 'conceptualized': 'conceptualised', 'conceptualizes': 'conceptualises', 
    'conceptualizing': 'conceptualising', 'connection': 'connexion', 'connections': 'connexions', 
    'contextualize': 'contextualise', 'contextualized': 'contextualised', 'contextualizes': 
    'contextualises', 'contextualizing': 'contextualising', 'councilor': 'councillor', 'councilors': 
    'councillors', 'counseled': 'counselled', 'counseling': 'counselling', 'counselor': 'counsellor', 
    'counselors': 'counsellors', 'cozier': 'cosier', 'cozies': 'cosies', 'coziest': 'cosiest', 'cozily': 
    'cosily', 'coziness': 'cosiness', 'cozy': 'cosy', 'crenelated': 'crenellated', 'criminalize': 
    'criminalise', 'criminalized': 'criminalised', 'criminalizes': 'criminalises', 'criminalizing': 
    'criminalising', 'criticize': 'criticise', 'criticized': 'criticised', 'criticizes': 'criticises', 
    'criticizing': 'criticising', 'crueler': 'crueller', 'cruelest': 'cruellest', 'crystallization': 
    'crystallisation', 'crystallize': 'crystallise', 'crystallized': 'crystallised', 'crystallizes': 
    'crystallises', 'crystallizing': 'crystallising', 'cudgeled': 'cudgelled', 'cudgeling': 
    'cudgelling', 'customize': 'customise', 'customized': 'customised', 'customizes': 'customises', 
    'customizing': 'customising', 'decentralization': 'decentralisation', 'decentralize': 
    'decentralise', 'decentralized': 'decentralised', 'decentralizes': 'decentralises', 
    'decentralizing': 'decentralising', 'decriminalization': 'decriminalisation', 'decriminalize': 
    'decriminalise', 'decriminalized': 'decriminalised', 'decriminalizes': 'decriminalises', 
    'decriminalizing': 'decriminalising', 'defense': 'defence', 'defenseless': 'defenceless', 
    'defenses': 'defences', 'dehumanization': 'dehumanisation', 'dehumanize': 'dehumanise', 
    'dehumanized': 'dehumanised', 'dehumanizes': 'dehumanises', 'dehumanizing': 'dehumanising', 
    'demeanor': 'demeanour', 'demilitarization': 'demilitarisation', 'demilitarize': 'demilitarise', 
    'demilitarized': 'demilitarised', 'demilitarizes': 'demilitarises', 'demilitarizing': 
    'demilitarising', 'demobilization': 'demobilisation', 'demobilize': 'demobilise', 'demobilized': 
    'demobilised', 'demobilizes': 'demobilises', 'demobilizing': 'demobilising', 'democratization': 
    'democratisation', 'democratize': 'democratise', 'democratized': 'democratised', 'democratizes': 
    'democratises', 'democratizing': 'democratising', 'demonize': 'demonise', 'demonized': 'demonised', 
    'demonizes': 'demonises', 'demonizing': 'demonising', 'demoralization': 'demoralisation', 
    'demoralize': 'demoralise', 'demoralized': 'demoralised', 'demoralizes': 'demoralises', 
    'demoralizing': 'demoralising', 'denationalization': 'denationalisation', 'denationalize': 
    'denationalise', 'denationalized': 'denationalised', 'denationalizes': 'denationalises', 
    'denationalizing': 'denationalising', 'deodorize': 'deodorise', 'deodorized': 'deodorised', 
    'deodorizes': 'deodorises', 'deodorizing': 'deodorising', 'depersonalize': 'depersonalise', 
    'depersonalized': 'depersonalised', 'depersonalizes': 'depersonalises', 'depersonalizing': 
    'depersonalising', 'deputize': 'deputise', 'deputized': 'deputised', 'deputizes': 'deputises', 
    'deputizing': 'deputising', 'desensitization': 'desensitisation', 'desensitize': 'desensitise', 
    'desensitized': 'desensitised', 'desensitizes': 'desensitises', 'desensitizing': 'desensitising', 
    'destabilization': 'destabilisation', 'destabilize': 'destabilise', 'destabilized': 'destabilised', 
    'destabilizes': 'destabilises', 'destabilizing': 'destabilising', 'dialed': 'dialled', 'dialing': 
    'dialling', 'dialog': 'dialogue', 'dialogs': 'dialogues', 'diarrhea': 'diarrhoea', 'digitize': 
    'digitise', 'digitized': 'digitised', 'digitizes': 'digitises', 'digitizing': 'digitising', 
    'discolor': 'discolour', 'discolored': 'discoloured', 'discoloring': 'discolouring', 'discolors': 
    'discolours', 'disemboweled': 'disembowelled', 'disemboweling': 'disembowelling', 'disfavor': 
    'disfavour', 'disheveled': 'dishevelled', 'passivizes': 'passivises', 'passivizing': 'passivising', 
    'pasteurization': 'pasteurisation', 'pasteurize': 'pasteurise', 'pasteurized': 'pasteurised', 
    'pasteurizes': 'pasteurises', 'pasteurizing': 'pasteurising', 'patronize': 'patronise', 
    'patronized': 'patronised', 'patronizes': 'patronises', 'patronizing': 'patronising', 
    'patronizingly': 'patronisingly', 'pedaled': 'pedalled', 'pedaling': 'pedalling', 'pederast': 
    'paederast', 'pederasts': 'paederasts', 'pedestrianization': 'pedestrianisation', 'pedestrianize': 
    'pedestrianise', 'pedestrianized': 'pedestrianised', 'pedestrianizes': 'pedestrianises', 
    'pedestrianizing': 'pedestrianising', 'pediatric': 'paediatric', 'pediatrician': 'paediatrician', 
    'pediatricians': 'paediatricians', 'pediatrics': 'paediatrics', 'pedophile': 'paedophile', 
    'pedophiles': 'paedophiles', 'pedophilia': 'paedophilia', 'penalize': 'penalise', 'penalized': 
    'penalised', 'penalizes': 'penalises', 'penalizing': 'penalising', 'penciled': 'pencilled', 
    'penciling': 'pencilling', 'personalize': 'personalise', 'personalized': 'personalised', 
    'personalizes': 'personalises', 'personalizing': 'personalising', 'pharmacopeia': 'pharmacopoeia', 
    'pharmacopeias': 'pharmacopoeias', 'philosophize': 'philosophise', 'philosophized': 'philosophised', 
    'philosophizes': 'philosophises', 'philosophizing': 'philosophising', 'phony ': 'phoney ', 
    'pizzazz': 'pzazz', 'plagiarize': 'plagiarise', 'plagiarized': 'plagiarised', 'plagiarizes': 
    'plagiarises', 'plagiarizing': 'plagiarising', 'plow': 'plough', 'plowed': 'ploughed', 'plowing': 
    'ploughing', 'plowman': 'ploughman', 'plowmen': 'ploughmen', 'plows': 'ploughs', 'plowshare': 
    'ploughshare', 'plowshares': 'ploughshares', 'polarization': 'polarisation', 'polarize': 'polarise', 
    'polarized': 'polarised', 'polarizes': 'polarises', 'polarizing': 'polarising', 'politicization': 
    'politicisation', 'politicize': 'politicise', 'politicized': 'politicised', 'politicizes': 
    'politicises', 'politicizing': 'politicising', 'popularization': 'popularisation', 'popularize': 
    'popularise', 'popularized': 'popularised', 'popularizes': 'popularises', 'popularizing': 
    'popularising', 'pouf': 'pouffe', 'poufs': 'pouffes', 'practice': 'practise', 'practiced': 
    'practised', 'practices': 'practises', 'practicing ': 'practising ', 'presidium': 'praesidium', 
    'presidiums ': 'praesidiums ', 'pressurization': 'pressurisation', 'pressurize': 'pressurise', 
    'pressurized': 'pressurised', 'pressurizes': 'pressurises', 'pressurizing': 'pressurising', 
    'pretense': 'pretence', 'pretenses': 'pretences', 'primeval': 'primaeval', 'prioritization': 
    'prioritisation', 'prioritize': 'prioritise', 'prioritized': 'prioritised', 'prioritizes': 
    'prioritises', 'prioritizing': 'prioritising', 'privatization': 'privatisation', 'privatizations': 
    'privatisations', 'privatize': 'privatise', 'privatized': 'privatised', 'privatizes': 'privatises', 
    'privatizing': 'privatising', 'professionalization': 'professionalisation', 'professionalize': 
    'professionalise', 'professionalized': 'professionalised', 'professionalizes': 'professionalises', 
    'professionalizing': 'professionalising', 'program': 'programme', 'programs': 'programmes', 
    'prolog': 'prologue', 'prologs': 'prologues', 'propagandize': 'propagandise', 'propagandized': 
    'propagandised', 'propagandizes': 'propagandises', 'propagandizing': 'propagandising', 
    'proselytize': 'proselytise', 'proselytized': 'proselytised', 'proselytizer': 'proselytiser', 
    'proselytizers': 'proselytisers', 'proselytizes': 'proselytises', 'proselytizing': 'proselytising', 
    'psychoanalyze': 'psychoanalyse', 'psychoanalyzed': 'psychoanalysed', 'psychoanalyzes': 
    'psychoanalyses', 'psychoanalyzing': 'psychoanalysing', 'publicize': 'publicise', 'publicized': 
    'publicised', 'publicizes': 'publicises', 'publicizing': 'publicising', 'pulverization': 
    'pulverisation', 'pulverize': 'pulverise', 'pulverized': 'pulverised', 'pulverizes': 'pulverises', 
    'pulverizing': 'pulverising', 'pummel': 'pummelled', 'pummeled': 'pummelling', 'quarreled': 
    'quarrelled', 'quarreling': 'quarrelling', 'radicalize': 'radicalise', 'radicalized': 'radicalised', 
    'radicalizes': 'radicalises', 'radicalizing': 'radicalising', 'rancor': 'rancour', 'randomize': 
    'randomise', 'randomized': 'randomised', 'randomizes': 'randomises', 'randomizing': 'randomising', 
    'rationalization': 'rationalisation', 'rationalizations': 'rationalisations', 'rationalize': 
    'rationalise', 'rationalized': 'rationalised', 'rationalizes': 'rationalises', 'rationalizing': 
    'rationalising', 'raveled': 'ravelled', 'raveling': 'ravelling', 'realizable': 'realisable', 
    'realization': 'realisation', 'realizations': 'realisations', 'realize': 'realise', 'realized': 
    'realised', 'realizes': 'realises', 'realizing': 'realising', 'recognizable': 'recognisable', 
    'recognizably': 'recognisably', 'recognizance': 'recognisance', 'recognize': 'recognise', 
    'recognized': 'recognised', 'recognizes': 'recognises', 'recognizing': 'recognising', 'reconnoiter': 
    'reconnoitre', 'reconnoitered': 'reconnoitred', 'reconnoitering': 'reconnoitring', 'reconnoiters': 
    'reconnoitres', 'refueled': 'refuelled', 'refueling': 'refuelling', 'regularization': 
    'regularisation', 'regularize': 'regularise', 'regularized': 'regularised', 'regularizes': 
    'regularises', 'regularizing': 'regularising', 'remodeled': 'remodelled', 'remodeling': 
    'remodelling', 'remold': 'remould', 'remolded': 'remoulded', 'remolding': 'remoulding', 'remolds': 
    'remoulds', 'reorganization': 'reorganisation', 'reorganizations': 'reorganisations', 'reorganize': 
    'reorganise', 'reorganized': 'reorganised', 'reorganizes': 'reorganises', 'reorganizing': 
    'reorganising', 'reveled': 'revelled', 'reveler': 'reveller', 'revelers': 'revellers', 'reveling': 
    'revelling', 'revitalize': 'revitalise', 'revitalized': 'revitalised', 'revitalizes': 'revitalises', 
    'revitalizing': 'revitalising', 'revolutionize': 'revolutionise', 'revolutionized': 
    'revolutionised', 'revolutionizes': 'revolutionises', 'revolutionizing': 'revolutionising', 
    'rhapsodize': 'rhapsodise', 'rhapsodized': 'rhapsodised', 'rhapsodizes': 'rhapsodises', 
    'rhapsodizing': 'rhapsodising', 'rigor': 'rigour', 'rigors': 'rigours', 'ritualized': 'ritualised', 
    'rivaled': 'rivalled', 'rivaling': 'rivalling', 'romanticize': 'romanticise', 'romanticized': 
    'romanticised', 'romanticizes': 'romanticises', 'romanticizing': 'romanticising', 'rumor': 'rumour', 
    'rumored': 'rumoured', 'rumors': 'rumours', 'saber': 'sabre', 'sabers': 'sabres', 'saltpeter': 
    'saltpetre', 'sanitize': 'sanitise', 'sanitized': 'sanitised', 'sanitizes': 'sanitises', 
    'sanitizing': 'sanitising', 'satirize': 'satirise', 'satirized': 'satirised', 'satirizes': 
    'satirises', 'satirizing': 'satirising', 'savior': 'saviour', 'saviors': 'saviours', 'savor': 
    'savour', 'savored': 'savoured', 'savories': 'savouries', 'savoring': 'savouring', 'savors': 
    'savours', 'savory': 'savoury', 'scandalize': 'scandalise', 'scandalized': 'scandalised', 
    'scandalizes': 'scandalises', 'scandalizing': 'scandalising', 'scepter': 'sceptre', 'scepters': 
    'sceptres', 'scrutinize': 'scrutinise', 'scrutinized': 'scrutinised', 'scrutinizes': 'scrutinises', 
    'scrutinizing': 'scrutinising', 'secularization': 'secularisation', 'secularize': 'secularise', 
    'secularized': 'secularised', 'secularizes': 'secularises', 'secularizing': 'secularising', 
    'sensationalize': 'sensationalise', 'sensationalized': 'sensationalised', 'sensationalizes': 
    'sensationalises', 'sensationalizing': 'sensationalising', 'sensitize': 'sensitise', 'sensitized': 
    'sensitised', 'sensitizes': 'sensitises', 'sensitizing': 'sensitising', 'sentimentalize': 
    'sentimentalise', 'sentimentalized': 'sentimentalised', 'sentimentalizes': 'sentimentalises', 
    'sentimentalizing': 'sentimentalising', 'sepulcher': 'sepulchre', 'sepulchers ': 'sepulchres', 
    'serialization': 'serialisation', 'serializations': 'serialisations', 'serialize': 'serialise', 
    'serialized': 'serialised', 'serializes': 'serialises', 'serializing': 'serialising', 'sermonize': 
    'sermonise', 'sermonized': 'sermonised', 'sermonizes': 'sermonises', 'sermonizing': 'sermonising', 
    'sheik ': 'sheikh ', 'shoveled': 'shovelled', 'shoveling': 'shovelling', 'shriveled': 'shrivelled', 
    'shriveling': 'shrivelling', 'signaled': 'signalled', 'signaling': 'signalling', 'signalize': 
    'signalise', 'signalized': 'signalised', 'signalizes': 'signalises', 'signalizing': 'signalising', 
    'siphon': 'syphon', 'siphoned': 'syphoned', 'siphoning': 'syphoning', 'siphons': 'syphons', 
    'skeptic': 'sceptic', 'skeptical': 'sceptical', 'skeptically': 'sceptically', 'skepticism': 
    'scepticism', 'skeptics': 'sceptics', 'smolder': 'smoulder', 'smoldered': 'smouldered', 
    'smoldering': 'smouldering', 'smolders': 'smoulders', 'sniveled': 'snivelled', 'sniveling': 
    'snivelling', 'snorkeled': 'snorkelled', 'snorkeling': 'snorkelling', 'snowplow': 'snowploughs', 
    'socialization': 'socialisation', 'socialize': 'socialise', 'socialized': 'socialised', 
    'socializes': 'socialises', 'socializing': 'socialising', 'sodomize': 'sodomise', 'sodomized': 
    'sodomised', 'sodomizes': 'sodomises', 'sodomizing': 'sodomising', 'solemnize': 'solemnise', 
    'solemnized': 'solemnised', 'solemnizes': 'solemnises', 'solemnizing': 'solemnising', 'somber': 
    'sombre', 'specialization': 'specialisation', 'specializations': 'specialisations', 'specialize': 
    'specialise', 'specialized': 'specialised', 'specializes': 'specialises', 'specializing': 
    'specialising', 'specter': 'spectre', 'specters': 'spectres', 'spiraled': 'spiralled', 'spiraling': 
    'spiralling', 'splendor': 'splendour', 'splendors': 'splendours', 'squirreled': 'squirrelled', 
    'squirreling': 'squirrelling', 'stabilization': 'stabilisation', 'stabilize': 'stabilise', 
    'stabilized': 'stabilised', 'stabilizer': 'stabiliser', 'stabilizers': 'stabilisers', 'stabilizes': 
    'stabilises', 'stabilizing': 'stabilising', 'standardization': 'standardisation', 'standardize': 
    'standardise', 'standardized': 'standardised', 'standardizes': 'standardises', 'standardizing': 
    'standardising', 'stenciled': 'stencilled', 'stenciling': 'stencilling', 'sterilization': 
    'sterilisation', 'sterilizations': 'sterilisations', 'sterilize': 'sterilise', 'sterilized': 
    'sterilised', 'sterilizer': 'steriliser', 'sterilizers': 'sterilisers', 'sterilizes': 'sterilises', 
    'sterilizing': 'sterilising', 'stigmatization': 'stigmatisation', 'stigmatize': 'stigmatise', 
    'stigmatized': 'stigmatised', 'stigmatizes': 'stigmatises', 'stigmatizing': 'stigmatising', 
    'stories': 'storeys', 'story': 'storey', 'subsidization': 'subsidisation', 'subsidize': 'subsidise', 
    'subsidized': 'subsidised', 'subsidizer': 'subsidiser', 'subsidizers': 'subsidisers', 'subsidizes': 
    'subsidises', 'subsidizing': 'subsidising', 'succor': 'succour', 'succored': 'succoured', 
    'succoring': 'succouring', 'succors': 'succours', 'sulfate': 'sulphate', 'sulfates': 'sulphates', 
    'sulfide': 'sulphide', 'sulfides': 'sulphides', 'sulfur': 'sulphur', 'sulfurous': 'sulphurous',
    'harbor':'harbour', 'utilise':'utilize','utilised':'utilized','utilising':'utilizing','utilises':'utilizes',
    }

gb2us = {v:k for k,v in us2gb.items()}