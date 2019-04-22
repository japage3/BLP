#import pandas as pd
import numpy as np
import csv
import random
import os
import datetime as dt
import pytz

def setPath():
    path = 'C:/Users/mekurish/Dropbox/class/ECON ML/hw/'
    return path

def setFileNames():
    names = {'characteristicsCSV': 'BLPTestChars.csv'}
    return names

def genParms():
    np.random.seed(11)
    random.seed(11)
    nMarkets = 100
    marketProbs = {'McDonalds': 0.85, 'BK': 0.75, 'Sweetgreen': 0.55, 'Wendy\'s': 0.55}
    itemDict = {'McDonalds': ["Quarter_Pounder", "McNuggets", 'Southwest_Salad'], 
                'BK': ['Whopper', 'Chicken_Tenders'],
                'Wendy\'s': ['Classic_Double', 'Wendys_Nuggets', 'Garden_Salad'],
                'Sweetgreen': ['Chicken_Bowl', 'Rustic_Salad']
                    }
    itemTypes = {"Quarter_Pounder": 'b', "McNuggets": 'c', 'Southwest_Salad': 's',
                  'Whopper': 'b', 'Chicken_Tenders': 'c',
                   'Classic_Double': 'b', 'Wendys_Nuggets': 'c', 'Garden_Salad': 's',
                    'Chicken_Bowl': 'c', 'Rustic_Salad': 's'}
    firms = ["McDonalds", "BK", "Sweetgreen", "Wendy's"]
    markets = range(0,nMarkets)
    characteristics = {}
    # I tested the ksais and market shares turned out to be extremely sensitive to them. Thus I picked some good looking baseline
    # values and then allowed them to vary market-by-market within a relatively narrow range.
    ksaiDict = {"Quarter_Pounder": 2, "McNuggets": 3.2, 'Southwest_Salad': -2.75,
                  'Whopper': 1.8, 'Chicken_Tenders': 2.75,
                   'Classic_Double': 3, 'Wendys_Nuggets': 2.3, 'Garden_Salad': -8.75,
                    'Chicken_Bowl': 4, 'Rustic_Salad': 0}
    # as with the xi values, so with prices. Shares are very sensitive, so it was best to test them out in advance and keep minimally
    # variable values
    priceDict = {"Quarter_Pounder": 2.2, "McNuggets": 1.8, 'Southwest_Salad': 1.6,
                  'Whopper': 3.0, 'Chicken_Tenders': 2,
                   'Classic_Double': 2, 'Wendys_Nuggets': 2, 'Garden_Salad': 1.5,
                    'Chicken_Bowl': 5, 'Rustic_Salad': 7}
    for firm in firms:
        characteristics[firm] = {}
        for market in markets:
            characteristics[firm][market] = {}
            for item in itemDict[firm]:
                iType = itemTypes[item]
                characteristics[firm][market][item]= {}
                if iType == 'b':
                    characteristics[firm][market][item]['protein'] = np.random.randint(17,21)
                    characteristics[firm][market][item]['fat'] = np.random.randint(32,37)
                elif iType == 'c':
                    characteristics[firm][market][item]['protein'] = np.random.randint(16,17)
                    characteristics[firm][market][item]['fat'] = np.random.randint(25,30)
                elif iType == 's':
                    characteristics[firm][market][item]['protein'] = np.random.randint(10,15)
                    characteristics[firm][market][item]['fat'] = np.random.randint(0,4)
                characteristics[firm][market][item]['price'] = priceDict[item] + np.random.uniform(-0.2,0.2)
                characteristics[firm][market][item]['ksai'] = ksaiDict[item] + np.random.uniform(-0.2,0.2)
    betas = [-0.5, 0.8, -0.2] # betas are for price, protein, fat - respectively
    sigmas = [0.2, 0.5, 0.5] 
    nus = [0, 0, 0]
    parameters = {'characteristics': characteristics, 'betas': betas, 'nus': nus, 'sigmas': sigmas, 
                    'nPerMarket': 500, 'nMarkets': nMarkets, 'mProbs': marketProbs}
    return parameters

def outChars(parms, path, names):
    fileName = names['characteristicsCSV']
    outFile = os.path.join(path, fileName)
    dta = parms['characteristics']
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        header = ['firm', 'market','product', 'price', 'protein', 'fat']
        writer.writerow(header)
        for f in dta:
            for market in dta[f]:
                for item in dta[f][market]:
                    row = [f]
                    row.append(market)
                    row.append(item)
                    row.append(dta[f][market][item]['price'])
                    row.append(dta[f][market][item]['protein'])
                    row.append(dta[f][market][item]['fat'])
                    writer.writerow(row)

def genPeople(parms, n):
    iden = 0
    markets = range(0, parms['nMarkets'])
    peopleBetas  = {}
    for i in markets:
        peopleBetas[i] = {}
        for j in range(0, n):
            iden += 1
            peopleBetas[i][iden] = {}
            for k, val in enumerate(parms['nus']):
                personalBeta = np.random.normal(parms['nus'][k], parms['sigmas'][k]) + parms['betas'][k]
                peopleBetas[i][iden][k] = personalBeta
    return peopleBetas

def genMarkets(parms):
    # goal is to randomly filter firms from markets so that we get diversity in firm entry into markets
    # 0 firm markets are not allowed - if a market is empty, we pick a random (not Wendy's for obvious reasons) firm to be in it
    num = parms['nMarkets']
    mProbs = parms['mProbs']
    marketFirms = {}
    for i in range(0, num):
        marketFirms[i] = {}
        counter = 0
        for firm in mProbs:
            rand = np.random.uniform(0,1)
            marketFirms[i][firm] = True if rand < mProbs[firm] else False 
            if rand < mProbs[firm]:
                counter += 1 
        if counter == 0:
            rand = np.random.uniform(0,1)
            if rand < 0.33:
                marketFirms[i]['McDonalds'] = True
            elif rand < 0.66:
                marketFirms[i]['BK'] = True
            else:
                marketFirms[i]['Sweetgreen'] = True
    return marketFirms

def simulateChoices(parms, people, markets, dave=True): 
    choices = {}
    for i, mark in enumerate(people):
        choices[mark] = {}          
        for j, person in enumerate(people[i]):
            utilities = []
            products = []
            for resto in parms['characteristics'].keys():
                if (resto != 'Wendy\'s' or dave) and markets[mark][resto]:
                    for prod in parms['characteristics'][resto][mark]:
                        price = parms['characteristics'][resto][mark][prod]['price']
                        protein = parms['characteristics'][resto][mark][prod]['protein']
                        fat = parms['characteristics'][resto][mark][prod]['fat']
                        ksai = parms['characteristics'][resto][mark][prod]['ksai']
                        u = people[mark][person][0]*price + people[mark][person][1]*protein + people[mark][person][2]*fat + ksai + np.random.gumbel()
                        utilities.append(u)
                        products.append(prod)
            # adding the outside option
            utilities.append(0)
            products.append('outside')
            choice = np.argmax(utilities)
            choices[mark][person] = products[choice]
    return choices
    
def marketShares(parms, choices, markets, dave=True):
    shares = {"overall": {}, "market": {}}
    nOverall = 0
    prodSet = set()
    for f in parms['characteristics']:
        for m in parms['characteristics'][f]:
            for p in parms['characteristics'][f][m]:
                prodSet.add((f, p))
    tOverall = {}
    for pair in prodSet:
        if pair[0] != 'Wendy\'s' or dave:
            tOverall[pair[1]] = 0
    for market in choices:
        nMarket = 0
        tMarket = {}
        for pair in prodSet:
            if pair[0] != 'Wendy\'s' or dave:
                tMarket[pair[1]] = 0
        for iden in choices[market]:
            c = choices[market][iden]
            if c == 'outside':
                continue
            tOverall[c] += 1
            tMarket[c] += 1
            nOverall += 1
            nMarket +=1
        for prod in tMarket:
            tMarket[prod] = float(tMarket[prod])/float(nMarket)
        shares['market'][market] = tMarket
    for prod in tOverall:
        tOverall[prod] = float(tOverall[prod])/float(nOverall)
    shares['overall'] = tOverall
    return shares  

def sharesAsCSV(shares, path, fileName):
    outFile = os.path.join(path, fileName)
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        if len(shares['overall'].keys()) == 3:
            header = ['Market', 'McDonalds', 'BK', 'Sweetgreen']
            firstRow = ['Total']
            for place in shares['overall']:
                firstRow.append('{0:.3g}'.format(shares['overall'][place]))
        else:
            header = ['Market', 'McDonalds', 'BK', 'Sweetgreen', 'Wendy\'s']
            firstRow = ['Total']
            for place in shares['overall']:
                firstRow.append('{0:.3g}'.format(shares['overall'][place]))
        writer.writerow(header)
        writer.writerow(firstRow)
        for item in shares['market']:
            row = [item]
            for place in shares['market'][item]:
                row.append('{0:.3g}'.format(shares['market'][item][place]))
            writer.writerow(row)

def generateEstimationData(shares, parms, path, fileName):
    outFile = os.path.join(path, fileName)
    firmCodes = {'McDonalds': 0, 'BK': 1, 'Sweetgreen': 2, 'Wendy\'s': 3}
    prodCodes = {"Quarter_Pounder": 0, "McNuggets": 1, 'Southwest_Salad': 2,
                  'Whopper': 3, 'Chicken_Tenders': 4,
                   'Classic_Double': 5, 'Wendys_Nuggets': 6, 'Garden_Salad': 7,
                    'Chicken_Bowl': 8, 'Rustic_Salad': 9}
    reverseLookup = {"Quarter_Pounder": 'McDonalds', "McNuggets": 'McDonalds', 'Southwest_Salad': 'McDonalds',
                  'Whopper': 'BK', 'Chicken_Tenders': 'BK',
                   'Classic_Double': 'Wendy\'s', 'Wendys_Nuggets': 'Wendy\'s', 'Garden_Salad': 'Wendy\'s',
                    'Chicken_Bowl': 'Sweetgreen', 'Rustic_Salad': 'Sweetgreen'}
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        header = ['market_ids','firm_ids', 'product_ids', 'shares', 'prices', 'protein', 'fat']
        writer.writerow(header)
        for market in shares['market']:
            for prod in shares['market'][market]:
                row = [market]
                firm = reverseLookup[prod]
                row.append(firmCodes[firm])
                row.append(prodCodes[prod])
                row.append(shares['market'][market][prod])
                row.append(parms['characteristics'][firm][market][prod]['price'])
                row.append(parms['characteristics'][firm][market][prod]['protein'])
                row.append(parms['characteristics'][firm][market][prod]['fat'])
                writer.writerow(row)

def main():
    start = dt.datetime.now(pytz.utc)
    path = setPath()
    fns = setFileNames()
    parms = genParms()
    outChars(parms, path, fns)
    # need to gen prices endogenously *after* the markets are simulated 
    people = genPeople(parms, parms['nPerMarket'])
    # randomizes which firms are present in each market
    markets = genMarkets(parms)
    # 'dave' is the Wendy's parameter, to include wendy's products in the choice set or not
    dave = False
    choices = simulateChoices(parms, people, markets, dave=dave)
    shares = marketShares(parms, choices, markets, dave=dave)
    #print(shares['overall'])
    dave = True
    newChoices = simulateChoices(parms, people, markets, dave=dave)
    newShares = marketShares(parms, newChoices, markets, dave=dave)
    path = setPath()
    #sharesAsCSV(shares, path, 'noWendys.csv')
    #sharesAsCSV(newShares, path, 'withWendys.csv')
    #choicesAsCSV(choices, parms, path, 'noWendysChoices.csv')
    #choicesAsCSV(newChoices, newParms, path, 'withWendysChoices.csv')
    generateEstimationData(newShares, parms, path, 'withWendysBLPData.csv')
    print('generated {} markets each with {} observations in {} minutes'.format(parms['nMarkets'], parms['nPerMarket'], \
    (dt.datetime.now(pytz.utc) - start).seconds/60))



if __name__ == '__main__':
    main()