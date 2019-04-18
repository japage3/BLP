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
    ksais = {}
    # not gonna let this vary by market
    ksais['outside'] = 0
    for firm in firms:
        characteristics[firm] = {}
        for item in itemDict[firm]:
            iType = itemTypes[item]
            # price is always set to 1 since it will be determined endogenously in a later method (which?)
            characteristics[firm][item]= {}
            characteristics[firm][item]['price'] = 1
            if iType == 'b':
                characteristics[firm][item]['protein'] = np.random.randint(10,20)
                characteristics[firm][item]['salt'] = np.random.randint(15,25)
                characteristics[firm][item]['fat'] = np.random.randint(35,45)
                characteristics[firm][item]['veggies'] = 0
            elif iType == 'c':
                characteristics[firm][item]['protein'] = np.random.randint(10,20)
                characteristics[firm][item]['salt'] = np.random.randint(15,25)
                characteristics[firm][item]['fat'] = np.random.randint(10,20)
                characteristics[firm][item]['veggies'] =  0
            elif iType == 's':
                characteristics[firm][item]['protein'] = np.random.randint(5,10)
                characteristics[firm][item]['salt'] = np.random.randint(15,25)
                characteristics[firm][item]['fat'] = np.random.randint(0,5)
                characteristics[firm][item]['veggies'] =  np.random.randint(2,3)
    # I tested the ksais and market shares turned out to be extremely sensitive to them. Thus I picked some good looking baseline
    # values and then allowed them to vary market-by-market within a relatively narrow range.
    ksaiDict = {"Quarter_Pounder": , "McNuggets": 'c', 'Southwest_Salad': 's',
                  'Whopper': 'b', 'Chicken_Tenders': 'c',
                   'Classic_Double': 'b', 'Wendys_Nuggets': 'c', 'Garden_Salad': 's',
                    'Chicken_Bowl': 'c', 'Rustic_Salad': 's'}
    for market in markets:
        ksais['outside'][market] = np.random.randint(0,10)
    betas = [1.0, -0.45, -0.15]
    sigmas = [1.5,0.6,0.6]
    nus = []
    for i in range(0,3):
        nus.append(np.random.normal(0,.2))

    parameters = {'characteristics': characteristics, 'ksais':ksais, 'betas': betas, 'price': -2.5, 'nus': nus, 'sigmas': sigmas, 'nPerMarket': 10000, 'numMarkets':nMarkets}
    return parameters

def outChars(parms, path, names):
    fileName = names['characteristicsCSV']
    outFile = os.path.join(path, fileName)
    dta = parms['characteristics']
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        header = ['firm', 'product', 'price', 'protein', 'fat', 'salt', 'veggies']
        writer.writerow(header)
        for f in dta:
            for item in dta[f]:
                row = [f]
                row.append(item)
                row.append(dta[f][item]['price'])
                row.append(dta[f][item]['protein'])
                row.append(dta[f][item]['fat'])
                row.append(dta[f][item]['salt'])
                row.append(dta[f][item]['veggies'])
                writer.writerow(row)



def addWendys(parms):
    chars = parms['characteristics']
    ksais = parms['ksais']
    chars['Wendys'] = {}
    ksais['Wendys'] = {}
    for market in chars['McDonalds']:
        chars['Wendys'][market] = {}
        chars['Wendys'][market]['price'] = np.random.randint(2,4)
        chars['Wendys'][market]['protein'] = np.random.randint(15,25)
        chars['Wendys'][market]['salt'] = np.random.uniform(30,50)
        chars['Wendys'][market]['fat'] = np.random.uniform(5,10)
        ksais['Wendys'][market] = np.random.normal(3,1)
    parameters = {'characteristics': chars, 'ksais':ksais, 'betas': parms['betas'], 'price': parms['price'], 'nus': parms['nus'], 'sigmas': parms['sigmas']}
    return parameters

def genPeople(parms, n):
    iden = 0
    markets = range(0, parms['numMarkets'])
    peopleBetas  = {}
    for i in markets:
        peopleBetas[i] = []
        for j in range(0, n):
            iden += 1
            indBetas = [iden, parms['price']]
            for k, val in enumerate(parms['nus']):
                indBetas.append(np.random.normal(parms['nus'][k], parms['sigmas'][k]) + parms['betas'][k])
            peopleBetas[i].append(indBetas)
    return peopleBetas

def simulateChoices(parms, people): 
    choices = {}
    for i, mark in enumerate(people):
        choices[mark] = {}          
        for j, person in enumerate(people[i]):
            utilities = []
            for resto in parms['characteristics'].keys():
                price = parms['characteristics'][resto][mark]['price']
                protein = parms['characteristics'][resto][mark]['protein']
                salt = parms['characteristics'][resto][mark]['salt']
                fat = parms['characteristics'][resto][mark]['fat']
                ksai = parms['ksais'][resto][mark] 
                u = parms['price']*price + people[mark][j][2]*protein + people[mark][j][3]*salt + people[mark][j][4]*fat + ksai + np.random.gumbel()
              #  print([price, people[mark][j][2], protein, people[mark][j][3], salt, people[mark][j][4], fat, ksai, 10*np.random.gumbel()])
                utilities.append(u)
            outside = parms['ksais']['outside'][mark]
            utilities.append(outside)
            #if j > 2:
             #  assert False
            choice = np.argmax(utilities)
            choices[mark][people[mark][j][0]] = choice 
    return choices
    
def marketShares(choices, wendys=False):
    shares = {"overall": {}, "market": {}}
    nOverall = 0
    if wendys:
        tOverall = {'McDonalds': 0, 'BK': 0, 'Sweetgreen': 0, 'Wendys': 0}
    else:
        tOverall = {'McDonalds': 0, 'BK': 0, 'Sweetgreen': 0}
    for market in choices:
        nMarket = 0
        if wendys:
            tMarket = {'McDonalds': 0, 'BK': 0, 'Sweetgreen': 0, 'Wendys': 0}
        else:
            tMarket = {'McDonalds': 0, 'BK': 0, 'Sweetgreen': 0}
        for iden in choices[market]:
            c = choices[market][iden]
            if c == 0:
                tOverall['McDonalds'] += 1
                tMarket['McDonalds'] += 1
            elif c == 1:
                tOverall['BK'] += 1
                tMarket['BK'] += 1
            elif c == 2:
                tOverall['Sweetgreen'] += 1
                tMarket['Sweetgreen'] += 1
            elif c == 3 and wendys:
                tOverall['Wendys'] += 1
                tMarket['Wendys'] += 1
            nOverall += 1
            nMarket +=1
        for store in tMarket:
            tMarket[store] = float(tMarket[store])/float(nMarket)
        shares['market'][market] = tMarket
    for store in tOverall:
        tOverall[store] = float(tOverall[store])/float(nOverall)
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
    revFirmLookup = {'McDonalds': 0, 'BK': 1, 'Sweetgreen': 2, 'Wendys': 3}
    pLookup = {'McDonalds': 0, 'BK': 1, 'Sweetgreen': 2, 'Wendys': 3}
    with open(outFile, 'w+', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='\"')
        header = ['market_ids','firm_ids', 'product_ids', 'shares', 'prices', 'protein', 'fat', 'salt']
        writer.writerow(header)
        for market in shares['market']:
            for prod in shares['market'][market]:
                row = [market]
                row.append(revFirmLookup[prod])
                row.append(pLookup[prod])
                row.append(shares['market'][market][prod])
                row.append(parms['characteristics'][prod][market]['price'])
                row.append(parms['characteristics'][prod][market]['protein'])
                row.append(parms['characteristics'][prod][market]['fat'])
                row.append(parms['characteristics'][prod][market]['salt'])
                writer.writerow(row)


def main():
    start = dt.datetime.now(pytz.utc)
    path = setPath()
    fns = setFileNames()
    parms = genParms()
    # check line to output parms to excel, should be commented out in the real run
    outChars(parms, path, fns)
    assert False
    # need to gen prices endogenously *after* the markets are simulated 
    people = genPeople(parms, parms['nPerMarket'])
    choices = simulateChoices(parms, people)
    shares = marketShares(choices)
    print(shares['overall'])
    assert False
    newParms = genParms()
    newChoices = simulateChoices(newParms, people)
    newShares = marketShares(newChoices, wendys=True)
    print(newShares['overall'])
    # print(newShares['overall'])
    path = setPath()
    sharesAsCSV(shares, path, 'noWendys.csv')
    sharesAsCSV(newShares, path, 'withWendys.csv')
    #choicesAsCSV(choices, parms, path, 'noWendysChoices.csv')
    #choicesAsCSV(newChoices, newParms, path, 'withWendysChoices.csv')
    generateEstimationData(newShares, newParms, path, 'withWendysBLPData.csv')
    print('generated {} markets each with {} observations in {} minutes'.format(parms['numMarkets'], parms['nPerMarket'], \
        (dt.datetime.now(pytz.utc) - start).seconds/60))



if __name__ == '__main__':
    main()