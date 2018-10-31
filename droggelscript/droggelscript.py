#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:56:35 2018

@author: sunshower
"""

import numpy as np

# logische Funktionen angewandt auf eine Liste mit
# Wahrheitswerten
def left_hand_side_aufgabe_a(a):
    return not(a[0] and a[1] and not(a[2]))

def left_hand_side_aufgabe_b(a):
    return (not(a[0]) == (not(a[2]) or a[1]))

def right_hand_side_aufgabe_b(a):
    return((not(a[1] or a[0])) or a[2])
    
def right_hand_side_aufgabe_a(a):
    return(not(not(a[0]) or (not(a[2]) or a[1])) or (not(a[0] and not(a[1]))))
    
def equivalent(a):
    return(a[0] == a[1])

# Anzahl logischer Variablen
n = 3

# Alle Belegungen der logischen Variablen erzeugen
V = np.unpackbits(np.array(range(0,2**n), dtype=np.uint8).reshape(2**n,1),axis=1)[:,-n:]

# logische Funktionen zeilenweise auswerten: die Ergebnisse stehen
# in den Spalten n+1, n+2, ...
V = np.hstack((V, np.apply_along_axis(left_hand_side_aufgabe_a, 1, V).reshape(2**n,1)))
V = np.hstack((V, np.apply_along_axis(right_hand_side_aufgabe_a, 1, V).reshape(2**n,1)))
V = np.hstack((V, np.apply_along_axis(equivalent, 1, V[:,-2:]).reshape(2**n,1)))
# V enthält jetzt in der letzten Spalte die Antwort: wenn alles Einsen sind
# sind die rechte und linke Seite äquivalent - andernfalls nicht; dort, 
# wo Nullen stehen, wird angezeigt, wo die linke und rechte Seite 
# verschiedene Antworten liefern

print('Aufgabe a: \n'+str(V))

# nimm wieder nur die ursprünglichen Belegungen der Variablen
V = V[:,:3]

V = np.hstack((V, np.apply_along_axis(left_hand_side_aufgabe_b, 1, V).reshape(2**n,1)))
V = np.hstack((V, np.apply_along_axis(right_hand_side_aufgabe_b, 1, V).reshape(2**n,1)))
V = np.hstack((V, np.apply_along_axis(equivalent, 1, V[:,-2:]).reshape(2**n,1)))

print('\nAufgabe b: \n'+str(V))
