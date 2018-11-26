import hashlib
import hmac

import sys

def seed2hpriv(seed):
    master_secret= seed.encode('utf-8')
    hk=hmac.HMAC(key=b"Bitcoin seed", msg=master_secret, digestmod=hashlib.sha512)  
    hp=hk.hexdigest()[0:64]
    return (hp)

seed = 'awake book subject inch gentle blur grant damage process float month clown'
master_secret= seed.encode('utf-8')

hk=hmac.HMAC(key=b"Bitcoin seed", msg=master_secret, digestmod=hashlib.sha512)  
hp=hk.hexdigest()[0:64]
print (seed)
print (hp)


#Below generate correct public address as found with money

import hashlib
from pycoin import ecdsa, encoding
import os
import codecs


if __name__ == '__main__':

    #rand = codecs.encode(os.urandom(32), 'hex').decode()
    #secret_exponent= int('0x'+rand, 0)

    secret_exponent = int( hp, 16 )

    print ('WIF: ' + encoding.secret_exponent_to_wif(secret_exponent, compressed=False))
    public_pair = ecdsa.public_pair_for_secret_exponent(ecdsa.secp256k1.generator_secp256k1, secret_exponent)
    hash160 = encoding.public_pair_to_hash160_sec(public_pair, compressed=True)

    codecs.encode(hash160,'hex')

    print('Bitcoin address: %s' % encoding.hash160_sec_to_bitcoin_address(hash160))

    with open('../../Mining/dict/UrbanDictionary.txt','r') as f:
	    contents = f.readlines()

    fout = open('privaddr-pair.txt', 'w' )

    i = 0
    for l in contents :
        priv = seed2hpriv( l )
        secret_exponent = int( priv, 16 )
        wif = encoding.secret_exponent_to_wif(secret_exponent, compressed=False)

        public_pair = ecdsa.public_pair_for_secret_exponent(ecdsa.secp256k1.generator_secp256k1, secret_exponent)
        hash160 = encoding.public_pair_to_hash160_sec(public_pair, compressed=True)
        pub = codecs.encode(hash160,'hex')
        addr = encoding.hash160_sec_to_bitcoin_address(hash160)
        
        if i%100 == 0 :
            print ( 'ADDR/WIF', addr,wif )
        fout.write( "%s:%s:%s:%s" % ( addr, wif, priv, l ) )

        i = i + 1

    fout.close()

