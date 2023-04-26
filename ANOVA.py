#!/usr/bin/env python
# coding: utf-8

# Laboratorijska vježba 3 - Poređenje više sistema : ANOVA i kontrasti
# 
# Zadatak: Napisati program za korištenje ANOVA tehnike i tehnike kontrasta.
# 
# Metodologija: Zadatak se radi samostalno. U proizvoljnom programskom jeziku
# napisati program za izračunavanje parametara ANOVA testa uz dozvoljavanje unosa
# proizvoljnog broja alternativa i proizvoljnog broja mjerenja svake od njih. Poželjno je
# da program ima grafički interfejs. Dodatno, implementirati tehniku kontrasta svake
# dvije poređene alternative.
# 

# Rješenje:
# 
# U Excel fajlu (alternative.xlsx) je data tabela mjerenja i alternativa. Nakon učitavanja podataka iz tabele vršićemo izračunavanje parametara ANOVA testa po koracima po kojima smo to radili na laboratorijskim vježbama, nakon čega ćemo sprovesti i tehniku kontrasta.

# In[115]:


# Uvezimo prvo potrebne biblioteke
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t # za računanje kritične vrijednosti Studentove raspodjele
import seaborn as sns
from itertools import combinations # za dobijanje kombinacija parova alternativa


# In[118]:


# Čitanje .xlsx fajla (u kojem su zabilježena mjerenja alternativa) i prikaz njegovog sadržaja
df = pd.read_excel('./alternative.xlsx')
print("Matrica mjerenja po alternativama:")
df


# Prije same tehnike ANOVA čisto informativno pogledajmo grafički prikaz raspodjele mjerenja po alternativama.

# In[119]:


# Preoblikujmo dataframe df kako bismo omogućili da bude pogodan za "statsmodels" paket
# koji će nam omogućiti da lakše vidimo razlike alternativa
df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['Alt 1', 'Alt 2', 'Alt 3', 'Alt 4'])

# Premiještanje imena kolona
df_melt.columns = ['index', 'Alternatives', 'Value']

# Generišimo boxplot kako bismo vidjeli raspodjelu podataka po alternativama 
# Korištenjem boxplot-a lakše detektujemo razlike između različitih alternativa
print("Grafički prikaz raspodjele mjerenja po alternativama:")
ax = sns.boxplot(x='Alternatives', y='Value', data=df_melt, color='#99c2a2')
ax = sns.swarmplot(x="Alternatives", y="Value", data=df_melt, color='#7d0013')
plt.show()


# Već na osnovu prethodnog dijagrama nam je jasno da se alternative 1 i 3 najmanje razlikuju.

# ANOVA TEHNIKA:

# In[120]:


# Kreiranje ANOVA tabele i računanje parametara ANOVA tehnike
data = [['Alternatives', '', '', '', '', ''], ['Error', '', '', '', '', ''], ['Total', '', '', '', '', '']] 
ANOVA_table = pd.DataFrame(data, columns = ['Variation', 'Sum of squares', 'Deg freedom', 'Mean square', 'Computed F', 'Tabulated F']) 
ANOVA_table.set_index('Variation', inplace = True)

# Izračunaćemo prvo srednje vrijednosti kolona i to smjestiti u vektor y_col_ms
# Vektor y_col_ms ima onoliko elemenata koliko ima kolona, tj. alternativa (k)
y_col_ms = df.mean()
# Broj mjerenja n je broj vrsta matrice df
n = np.shape(df)[0]
# Broj alternativa k je broj kolona matrice df
k = np.shape(df)[1]
# Nađimo sada srednju vrijednost svih elemenata matrice df od k*n elemenata - y_total_ms
y_total_ms = np.average(df)

# Izračunavanje SSA i ažuriranje ANOVA tabele
SSA_vector = n * (y_col_ms - y_total_ms)**2 # vektor od k elemenata
SSA = SSA_vector.sum()
# Ažuriranje ANOVA tabele
ANOVA_table['Sum of squares']['Alternatives'] = SSA
#print(SSA)

# Izračunavanje SSE i ažuriranje ANOVA tabele
SSE_matrix = (df - y_col_ms)**2 # matrica od k*n elemenata
SSE_vector = SSE_matrix.sum() # vektor od k elemenata
SSE = SSE_vector.sum()
# Ažuriranje ANOVA tabele
ANOVA_table['Sum of squares']['Error'] = SSE

# Izračunavanje SST i ažuriranje ANOVA tabele
SST = SSA + SSE
# Ažuriranje ANOVA tabele
ANOVA_table['Sum of squares']['Total'] = SST

# Izračunavanje stepeni slobode za SSA, SSE i SST i ažuriranje ANOVA tabele
df_SSA = k - 1 # zbog k alternativa
df_SSE = k * (n - 1) # zbog k alternativa, svaka sa n-1 stepeni slobode
df_SST = k * n - 1 # df_SST = df_SSA + df_SSE
# Ažuriranje ANOVA tabele
ANOVA_table['Deg freedom']['Alternatives'] = df_SSA
ANOVA_table['Deg freedom']['Error'] = df_SSE
ANOVA_table['Deg freedom']['Total'] = df_SST

# Izračunavanje varijanse sume kvadrata (srednje kvadratne vrijednosti) i ažuriranje ANOVA tabele
S_a = SSA / df_SSA # varijansa alternativa
S_e = SSE / df_SSE # varijansa greške
# Ažuriranje ANOVA tabele
ANOVA_table['Mean square']['Alternatives'] = S_a
ANOVA_table['Mean square']['Error'] = S_e

# Izračunavanje vrijednosti F_computed i ažuriranje ANOVA tabele
F_computed = S_a / S_e
# Ažuriranje ANOVA tabele
ANOVA_table['Computed F']['Alternatives'] = F_computed

# Izračunavanje vrijednosti F_tabulated i ažuriranje ANOVA tabele
# Ako pretpostavimo 95%-tni interval povjerenja, tada je 1-alpha=0.95 => alpha = 0.05
# F_tabulated se dobija iz tabele za vrijednosti [1-alpha;(k-1),k*(n-1)]
# U našem slučaju te vrijednosti su [0.95;3,16]
alpha = 0.05
# Izračunavanje F_tabulated
# Za ovo izračunavanje koristićemo funkciju stats.f.ppf() koja računa kritičnu vrijednost F raspodjele
# sa k-1 = 3 i k*(n-1) = 16 stepeni slobode sa 95%-nim intervalom povjerenja
# Ova funkcija zapravo daje ekvivalentnu vrijednost onoj iz naše skripte
F_tabulated = stats.f.ppf(1-alpha, ANOVA_table['Deg freedom']['Alternatives'], ANOVA_table['Deg freedom']['Error'])
ANOVA_table['Tabulated F']['Alternatives'] = F_tabulated

print("ANOVA tabela:")
ANOVA_table


# In[121]:


# Zaključak korištenja F-testa za poređenje odnosa varijansi
# Pogledajmo najprije odnose SSA/SST i SSE/SST
# koji nam daju uticaje pojedinih izvora varijacija (alternative i grešaka) u ukupnoj varijaciji
ratio_a = SSA/SST # udio varijacije zbog razlika između alternativa u ukupnoj varijaciji
ratio_e = SSE/SST # udio varijacije zbog grešaka u mjerenjima u ukupnoj varijaciji
print("ZAKLJUČAK SPROVEDENE ANOVA TEHNIKE:")
print("--------------------------------------------------------------------------------------")
print(ratio_a * 100,"[%] ukupne varijacije u mjerenjima je zbog razlika između alternativa.")
print("--------------------------------------------------------------------------------------")
print(ratio_e * 100,"[%] ukupne varijacije u mjerenjima je zbog grešaka u mjerenjima.")
print("--------------------------------------------------------------------------------------")
conclusion = "Što znači da imamo 95%-tno povjerenje da razlike između alternativa nisu statistički značajne."
if ANOVA_table['Computed F']['Alternatives'] > ANOVA_table['Tabulated F']['Alternatives']:
    conclusion = "Što znači da imamo 95%-tno povjerenje da su razlike između alternativa statistički značajne."
print("Rezultat F-testa:")
print("F izračunato je:", ANOVA_table['Computed F']['Alternatives'], " i F tabelarno je:", ANOVA_table['Tabulated F']['Alternatives'])
print(conclusion)


# TEHNIKA KONTRASTA:

# Tehnika kontrasta nam daje informaciju o tome kako se pojedini parovi alternativa razlikuju, za razliku od ANOVA tehnike koja nam samo pokazuje da li postoji statistički značajna razlika između alternativa.

# In[122]:


# Pronađimo najprije vektor efekata alpha_effects
# Efekat svake kolone (alternative) dobijamo kao razliku srednje vrijednosti te kolone
# i ukupne srednje vrijednosti matrice df, y_col_ms i y_total_ms, respektivno, u našem slučaju
alpha_effects = y_col_ms - y_total_ms
print("Vektor efekata:")
alpha_effects


# In[123]:


# Pogledajmo sada sve moguće kombinacije parova alternativa čije ćemo intervale povjerenja računati
combs = list(combinations(df, 2))
print("Sve moguće kombinacije parova alternativa su:")
combs


# In[124]:


# Računanje kritične vrijednosti Studentove t raspodjele
# Pretpostavimo 90%-tni interval povjerenja, tada je alpha=0.1 => alpha/2 = 0.05
# pa je a = 1-alpha/2 = 1-0.05 = 0.95
a = 0.95
# Stepen slobode je k*(n-1)
deg_freedom = k * (n-1)
# Pa se kritična vrijednost Studentove t raspodjele (t_value) dobija pozivom funkcije ppf():
t_value = t.ppf(a, deg_freedom)
print("Kritična vrijednost Studentove t raspodjele je:")
t_value


# In[125]:


# Računanje vrijednosti standardne devijacije Sc
# Standardna devijacija Se je kvadratni korijen varijanse S_e
Se = math.sqrt(S_e)
# Vrijednost ispod korijena je uvijek (2)/(k*n)
sq_rt = (2)/(k*n)
Sc = Se * math.sqrt(sq_rt)
print("Standardna devijacija Sc je:")
Sc


# Računanje kontrasta i intervala povjerenja za svaku kombinaciju alternativa koje se porede:

# In[126]:


groups = [] # vektor stringova parova alternativa koje se porede
contrast = [] # vektor vrijednosti kontrasta parova alternativa
c1 = [] # vektor donjih vrijednosti intervala povjerenja
c2 = [] # vektor gornjih vrijednosti intervala povjerenja
for comb in combs:
    # Izračunaćemo kontrast c za svaki par kao linaernu kombinaciju efekata alternativa
    c = alpha_effects[comb[0]] - alpha_effects[comb[1]]
    ci_lower = c - (t_value * Sc) # donja granica intervala povjerenja
    ci_upper = c + (t_value * Sc) # gornja granica intervala povjerenja
    # Dodavanje informacija o tekućem kontrastu na kraj odgovarajućeg vektora
    groups.append(str(comb[0]) + ' : ' + str(comb[1]))
    contrast.append(c)
    c1.append(ci_lower)
    c2.append(ci_upper)


# Tabelarni prikaz rezultata dobijenih tehnikom kontrasta

# In[127]:


# Prikažimo dobijene rezultate tabelarno
result_table = pd.DataFrame({'groups': groups,
                          'c': contrast,
                          'c1': c1,
                          'c2': c2})
result_table


# Grafički prikaz intervala povjerenja

# In[128]:


# Prikažimo dobijene intervale povjerenja grafički radi bolje vizuelizacije
data_dict = {}
data_dict['groups'] = groups
data_dict['ci_lower'] = c1
data_dict['ci_upper'] = c2
dataset = pd.DataFrame(data_dict)
# Grafički prikaz intervala povjerenja svih parova alternativa koje smo poredili
# Svaki interval povjerenja prikazan je jednom horizontalnom linijom u xy ravni
for ci_lower,ci_upper,y in zip(dataset['ci_lower'],dataset['ci_upper'],range(len(dataset))):
    plt.plot((ci_lower,ci_upper),(y,y),'ro-',color='red')
plt.yticks(range(len(dataset)),list(dataset['groups']))
# Predstavimo isprekidano osu x = 0 radi lakše detekcije intervala povjerenja koji (ne) sadrži 0
plt.vlines( 0, -1, len(groups), linestyles='dashed' )


# ZAKLJUČAK SPROVEDENE TEHNIKE KONTRASTA:

# Ukoliko interval povjerenja sadrži 0 ne postoji statistički značajna razlika između alternativa uključenih u kontrast. A ako interval povjerenja ne sadrži 0 tada postoji statistički značajna razlika između alternativa.
# 
# Na osnovu tabelarnog prikaza dobijenih rezultata, kao i grafičkog prikaza intervala povjerenja lako je zaključiti koji interval povjerenja sadrži/ne sadrži 0, a samim tim i koje alternative se statistički značajno ne razlikuju/razlikuju.
# 
# Za trenutni skup alternativa i mjerenja možemo zaključiti da samo za alternative Alt 1 i Alt 3 ne postoji statistički značajna razlika, dok se svi drugi parovi alternativa statistički značajno razlikuju.
