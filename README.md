# Predviđanje potrošnje za 24h unapred

Na početku fajla izvršeno je importovanje biblioteka koje se koriste u kodu i
postavljeni su parametri grafika.

Zatim su učitani csv fajlovi u Data Framove. Za svaki do tri fajla kreiran
je poseban df.

Izvršeno je inicijalno upoznavanje sa podacima iz foldera 'Weather_Daily' i
'Weather_Hourly' tako što je provereno koje vrednosti sadrži fajl i koliko puta
se javljaju u navedenim vajlovima.

Podaci iz fajla Weather_Hourly podeljeni su u posebne df koji sadrže podatke o
temperaturi, oblačnosti i vetru.

Urađen je, zatim, statistička analiza podatka koji se nalaze u novoformiranim
dfovima u kojima je primećeno da vrednosti podataka ne odstupaju od realno
očekivanih vrednosti.

Svakako referenti podatka je potrošnja pa se pristupilo pripremi podataka na
osnovu dostupnih vrednosti potrošnje. Kako su mi na raspolaganju bili podaci za
opterećenje od 15. aprila 2013. godine na taj vremenski interval su dovedeni
podaci o temperaturi. Primetio sam da podaci o oblačnosti i vetru imaju puno
nedostajućih podataka i da su dostupne vrednosti počevši od kraja 2014. godine.
Zbog ovoga, ali i zbog poznatih činjenica da oblačnost i vetar direktno utiču na
temperaturu koja je poznata u celom vremenskom periodu odlučio sam da ove podatke
ne razmatram nadalje u ovoj prvoj verziji treniranja modela.

Nakon toga je urađena priprema podatka za korišćenje funkcije difference koja
je pokazala da u podacima za potrošnju nedostaju merenja za dan 09. januar 2018.
godine. Dok kod podataka za temperaturu nedostaje samo jedna satna vrednost.

Svi nedostajući satni vremenski odbirci popunjeni su funkcijom resample() uz odabir
parametra H, a vrednosti su popunjene prethodnom izmerenom vrednošću korišćenjem
funkcije ffill(). Ovo je u slučaju temperature prihvatljiva aproksimacija jer se
temperatura iz sata u sat ne menja značajno.

Za potrošnju sam odabrao da nedostajuće vrednosi za da 09.01 popunim identičnim  
izmerenom u danu 08.01. Ova aproksimacija nije najbolje moguće rešenje, ali je
s obzirom da je poznato da postoji periodičnost opterećenja na dnevnom nivou, a
ovo se može videti i na slici 1.

Na sličan način urađena je priprema podataka iz drugog fajla 'Weather_Daily'.
S obzirom da su podaci dati na dnevnom nivou funkcijom resample() je urađeno
kreiranje vremskih odbiraka sa periodom od jednog sata. Nove satne vrednosti su
iste za sve odbirke u toku dana. I u ovom vajlu je izostalo merenje za 3 dana
što je takođe rešeno ovom funkcijom..

Mali problem je nastao oko korišćenja funcije ffil() jer po difoltu ne kreira
za poslednji dostupan datum već završava sa prethodnim. To je promenljivo
parametrom 'closed', ali u mom slučaju nije radilo onako kako sam želeo, pa sam
ja dodao još jedan datum pre korišćenja funcije resample() i onda sam nakon toga
obrisao datum na kraju.

Pripremljen je zbirni dataFrame u kojem je na kraju izvršena gorepomenuta
aproksimacija potrošnje za nedostajuči 09. januar 2018. godine.

Takođe, već formiranom zbirnom dataFrame-u dodate su vrednosti koje će modelu
dati uvid u dnevnu i godišnju periodičnost promene opterećenja. Potvrda da su
godišnja i dnevna periodičnost najizraženija vidljivo je na slici 1 (dva najveća
pika). Ono što sam primetio na slici 1 je treći po veličini pik koji je na
frekvenciji manjoj od jednog dana. Pretpostvljajući da su ovo podaci nekog
industrijskog potrošača ovaj pik je logičan je označava periodičnost u okviru
radnih sati ovog industijskog postorojenja. Za neki budući rad bi bilo verovatno
zanimljivo proveriti uticaj ove periodičnosti na rezultate koje model pravi.

Nadalje su podaci podeljeni u trening, validacioni i test skup i urađena im je
normalizacija.

Sekcija 'Model prepare' odnosi se na definisanje klase sa neophodnim metodima
za pripremu, podelu, prikazivanje podataka i kreiranje datasetova pogodnih za
treniranje modela.
Takođe u toj sekciji definisana je vrednost epoha za treniranje modela i funcija
za kompaliranje i fitovanje modela.

U sekciji 'Multi-step model' dati su ulazni parametri za kreiranje prozora podataka.
Ulazna širina je postavljena na 24 ulaznih podataka (24h) i isto toliko za izlazne
podatke.

Odlučio sam se da treniram 5 različitih modela i da na kraju prikažem njihovu
metriku na jednog grafiku kako bi se napravilo poređenje među modelima.

Trenirani su Linearni model, Dense model, CNN, RNN i autoregressive RNN model.
Za model CNN ulazni prozor je 12 sample jer sam želeo da proverim kakve će model
dati rezultate kad se promeni veličina prozora. Rezultati su bolji nego sa
3 ili 5 sample-a).
Takođe, za Autoregressive RNN definisana je i klasa sa funcijama neophodnim za
treniranje istog u skladu sa uputstvima iz tutorijala.

Grafici za trenirane modele sa prikazanim stvarnim i predviđenim vrednostima,
prikazani su na slikama 2-6, dok je na slici 7 dat grafik sa metrikom modela.

Iz priloženog se vidi da najbolje rezultate daje RNN model.
