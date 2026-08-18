"""Interfaz web local para el CatBoost del escenario RandomizedSearch.

Ejecute: py 11_interfaz_catboost.py
Se abrirá el navegador. Detenga el servidor con Ctrl+C.
No usa Flask, Streamlit ni otro framework web.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread, Timer
import json
import webbrowser

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RUTA_MODELO = BASE / "modelos" / "catboost_03_search.joblib"
RUTA_TRAIN = BASE / "data" / "processed" / "train_raw.csv"
TARGET = "y"
NUMERICAS = {"edad", "es_fin_semana"}
# Estos campos pueden recibir una categoría nueva. Los valores del train se
# muestran como sugerencias, pero no limitan lo que el usuario puede escribir.
CATEGORICAS_ABIERTAS = {"provincia", "canton", "lugar", "nacionalidad"}

ETIQUETAS = {
    "zona": "Zona", "provincia": "Provincia", "canton": "Cantón",
    "area_hecho": "Área del hecho", "lugar": "Lugar",
    "tipo_lugar": "Tipo de lugar", "presunta_motivacion": "Presunta motivación",
    "presun_motiva_observada": "Motivación observada", "edad": "Edad",
    "sexo": "Sexo", "etnia": "Etnia", "estado_civil": "Estado civil",
    "nacionalidad": "Nacionalidad", "discapacidad": "Discapacidad",
    "mes": "Mes", "dia_semana": "Día de la semana",
    "es_fin_semana": "Fin de semana", "hora": "Hora",
    "franja_horaria": "Franja horaria",
}
GRUPOS = {
    "Ubicación del hecho": ["zona", "provincia", "canton", "area_hecho", "lugar", "tipo_lugar"],
    "Contexto": ["presunta_motivacion", "presun_motiva_observada"],
    "Datos de la persona": ["edad", "sexo", "etnia", "estado_civil", "nacionalidad", "discapacidad"],
    "Fecha y hora": ["mes", "dia_semana", "es_fin_semana", "hora", "franja_horaria"],
}

HTML = r'''<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Clasificador CatBoost</title>
<style>
:root{--ink:#172033;--muted:#64748b;--primary:#3157d5;--primary2:#743fd1;--bg:#f1f5fb;--card:#fff;--line:#dce4f1;--good:#0f9f72}
*{box-sizing:border-box}body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--ink);background:radial-gradient(circle at top left,#e8e7ff,transparent 35%),var(--bg)}
header{color:white;background:linear-gradient(125deg,#172b63,#4b35a6 65%,#7040b9);padding:46px max(24px,calc((100% - 1160px)/2));box-shadow:0 5px 20px #172b6330}
header small{letter-spacing:.14em;text-transform:uppercase;opacity:.75;font-weight:700}h1{font-size:clamp(1.8rem,4vw,2.65rem);margin:9px 0 7px}header p{margin:0;opacity:.82}
main{max-width:1160px;margin:30px auto;padding:0 20px 50px}.status{display:flex;align-items:center;gap:9px;color:var(--muted);margin-bottom:18px;font-size:.92rem}.dot{width:9px;height:9px;background:var(--good);border-radius:50%;box-shadow:0 0 0 4px #0f9f7220}
form{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:23px;box-shadow:0 10px 30px #2332570b}.card h2{font-size:1.08rem;margin:0 0 20px;display:flex;align-items:center;gap:9px}.card h2:before{content:"";width:5px;height:22px;border-radius:5px;background:linear-gradient(var(--primary),var(--primary2))}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:16px}label{display:flex;flex-direction:column;gap:7px;font-size:.82rem;font-weight:700;color:#41506a}select,input{width:100%;border:1px solid #cad5e5;border-radius:9px;padding:11px 12px;background:#fbfdff;color:var(--ink);font:inherit;outline:none;transition:.18s}select:focus,input:focus{border-color:var(--primary);box-shadow:0 0 0 3px #3157d51b;background:white}
.actions{grid-column:1/-1;display:flex;justify-content:center;gap:12px;margin:6px 0}button{border:0;border-radius:10px;padding:13px 25px;font-weight:700;font-size:.95rem;cursor:pointer}#submit{color:white;min-width:190px;background:linear-gradient(120deg,var(--primary),var(--primary2));box-shadow:0 8px 20px #443cc33a}#submit:disabled{opacity:.6;cursor:wait}.secondary{background:white;border:1px solid var(--line);color:#52617a}
#result{grid-column:1/-1;display:none}.prediction{font-size:1.5rem;color:var(--primary);font-weight:800;margin:4px 0 20px}.prob-row{display:grid;grid-template-columns:180px 1fr 65px;gap:12px;align-items:center;margin:11px 0;font-size:.88rem}.track{height:11px;background:#e9eef7;border-radius:10px;overflow:hidden}.bar{height:100%;border-radius:10px;background:linear-gradient(90deg,var(--primary),#8b55d9);transition:width .7s}.percent{text-align:right;font-weight:700}.notice{color:var(--muted);font-size:.78rem;border-top:1px solid var(--line);padding-top:15px;margin-top:20px}.error{color:#b42318;background:#fff2f0;border:1px solid #ffc9c2;border-radius:9px;padding:12px}
@media(max-width:850px){form{grid-template-columns:1fr}.fields{grid-template-columns:1fr}.prob-row{grid-template-columns:125px 1fr 55px}}
</style></head><body><header><small>Proyecto de inteligencia artificial</small><h1>Clasificación del tipo de arma</h1><p>Predicción con CatBoost optimizado mediante RandomizedSearchCV</p></header>
<main><div class="status"><span class="dot"></span>Modelo cargado y listo para predecir</div><p class="notice">Los campos Provincia, Cantón, Lugar y Nacionalidad ofrecen sugerencias, pero también permiten escribir valores nuevos.</p><form id="form"><div id="cards" style="display:contents"></div><div class="actions"><button type="reset" class="secondary">Restablecer</button><button id="submit" type="submit">Realizar predicción</button></div><section class="card" id="result"><h2>Resultado del modelo</h2><div id="resultBody"></div></section></form></main>
<script>
const form=document.querySelector('#form'),cards=document.querySelector('#cards'),result=document.querySelector('#result');
function field(name,s){
 const label=document.createElement('label');label.textContent=s.label;let el;
 if(s.type==='number'){
  el=document.createElement('input');el.type='number';el.min=s.min;el.max=s.max;el.step=1;el.value=s.default
 }else if(s.type==='suggest'){
  el=document.createElement('input');el.type='text';el.value=s.default;el.placeholder='Seleccione o escriba un valor nuevo';
  const list=document.createElement('datalist');list.id=`list-${name}`;
  s.options.forEach(o=>{const op=document.createElement('option');op.value=o.value;list.append(op)});
  el.setAttribute('list',list.id);label.append(list)
 }else{
  el=document.createElement('select');s.options.forEach(o=>{const op=document.createElement('option');op.value=o.value;op.textContent=o.label;op.selected=o.value==s.default;el.append(op)})
 }
 el.name=name;el.required=true;label.append(el);return label
}
fetch('/config').then(r=>r.json()).then(data=>{Object.entries(data.groups).forEach(([title,names])=>{const card=document.createElement('section');card.className='card';card.innerHTML=`<h2>${title}</h2>`;const fields=document.createElement('div');fields.className='fields';names.forEach(n=>fields.append(field(n,data.fields[n])));card.append(fields);cards.append(card)})}).catch(()=>cards.innerHTML='<p class="error">No se pudo cargar la configuración.</p>');
form.addEventListener('submit',async e=>{e.preventDefault();const btn=document.querySelector('#submit');btn.disabled=true;btn.textContent='Analizando…';try{const data=Object.fromEntries(new FormData(form));const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const out=await response.json();if(!response.ok)throw new Error(out.error||'Error de predicción');document.querySelector('#resultBody').innerHTML=`<div class="prediction">${out.prediction}</div>`+out.probabilities.map(x=>`<div class="prob-row"><span>${x.class}</span><div class="track"><div class="bar" style="width:${x.probability*100}%"></div></div><span class="percent">${(x.probability*100).toFixed(2)}%</span></div>`).join('')+`<p class="notice">Esta salida es una estimación estadística y no debe usarse como único criterio para tomar decisiones.</p>`;result.style.display='block';result.scrollIntoView({behavior:'smooth'})}catch(err){document.querySelector('#resultBody').innerHTML=`<p class="error">${err.message}</p>`;result.style.display='block'}finally{btn.disabled=false;btn.textContent='Realizar predicción'}});form.addEventListener('reset',()=>result.style.display='none');
</script></body></html>'''


def cargar_recursos():
    faltantes = [p for p in (RUTA_MODELO, RUTA_TRAIN) if not p.exists()]
    if faltantes:
        raise FileNotFoundError("Faltan archivos:\n" + "\n".join(map(str, faltantes)))
    modelo = joblib.load(RUTA_MODELO)
    train = pd.read_csv(RUTA_TRAIN)
    return modelo, train, [c for c in train.columns if c != TARGET]


MODELO, TRAIN, COLUMNAS = cargar_recursos()
LOCK_MODELO = Lock()


def moda(serie):
    valores = serie.dropna().mode()
    return valores.iloc[0] if not valores.empty else "DESCONOCIDO"


def crear_configuracion():
    fields = {}
    for col in COLUMNAS:
        if col == "edad":
            mediana = float(pd.to_numeric(TRAIN[col], errors="coerce").median())
            fields[col] = {"label": ETIQUETAS[col], "type": "number", "min": 0, "max": 120, "default": round(mediana)}
        elif col == "es_fin_semana":
            fields[col] = {"label": ETIQUETAS[col], "type": "select", "options": [{"value": "0", "label": "No"}, {"value": "1", "label": "Sí"}], "default": str(int(moda(TRAIN[col])))}
        else:
            valores = sorted(TRAIN[col].fillna("DESCONOCIDO").astype(str).unique())
            tipo = "suggest" if col in CATEGORICAS_ABIERTAS else "select"
            fields[col] = {"label": ETIQUETAS.get(col, col), "type": tipo, "options": [{"value": v, "label": v} for v in valores], "default": str(moda(TRAIN[col].fillna("DESCONOCIDO")))}
    return {"groups": GRUPOS, "fields": fields}


CONFIG = crear_configuracion()


class Manejador(BaseHTTPRequestHandler):
    def responder(self, contenido, codigo=200, tipo="application/json; charset=utf-8"):
        datos = contenido.encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.responder(HTML, tipo="text/html; charset=utf-8")
        elif self.path == "/config":
            self.responder(json.dumps(CONFIG, ensure_ascii=False))
        else:
            self.responder(json.dumps({"error": "Ruta no encontrada"}), 404)

    def do_POST(self):
        if self.path != "/predict":
            self.responder(json.dumps({"error": "Ruta no encontrada"}), 404)
            return
        try:
            longitud = int(self.headers.get("Content-Length", 0))
            datos = json.loads(self.rfile.read(longitud).decode("utf-8"))
            faltantes = [c for c in COLUMNAS if c not in datos or datos[c] == ""]
            if faltantes:
                raise ValueError("Complete todos los campos: " + ", ".join(faltantes))
            fila = {c: datos[c] for c in COLUMNAS}
            fila["edad"] = float(fila["edad"])
            if not 0 <= fila["edad"] <= 120:
                raise ValueError("La edad debe estar entre 0 y 120.")
            fila["es_fin_semana"] = int(fila["es_fin_semana"])
            entrada = pd.DataFrame([fila], columns=COLUMNAS)
            for col in (c for c in COLUMNAS if c not in NUMERICAS):
                entrada[col] = entrada[col].astype(str).str.strip().str.upper()
                if entrada.at[0, col] == "":
                    raise ValueError(f"El campo {ETIQUETAS.get(col, col)} no puede estar vacío.")
            with LOCK_MODELO:
                pred = str(np.asarray(MODELO.predict(entrada)).reshape(-1)[0])
                prob = MODELO.predict_proba(entrada)[0]
            orden = np.argsort(prob)[::-1]
            respuesta = {"prediction": pred, "probabilities": [{"class": str(MODELO.classes_[i]), "probability": float(prob[i])} for i in orden]}
            self.responder(json.dumps(respuesta, ensure_ascii=False))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.responder(json.dumps({"error": str(exc)}, ensure_ascii=False), 400)
        except Exception as exc:
            self.responder(json.dumps({"error": f"No se pudo predecir: {exc}"}, ensure_ascii=False), 500)

    def log_message(self, formato, *args):
        return


def main():
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), Manejador)
    url = f"http://127.0.0.1:{servidor.server_address[1]}"
    print("\nInterfaz CatBoost iniciada correctamente.")
    try:
        from google.colab import output
    except ImportError:
        output = None

    # En Colab, localhost pertenece a la máquina remota. El proxy oficial de
    # Colab presenta el puerto dentro de una salida del notebook.
    if output is not None:
        print("Mostrando la interfaz dentro de Google Colab.\n")
        Thread(target=servidor.serve_forever, daemon=True).start()
        output.serve_kernel_port_as_iframe(
            servidor.server_address[1], width="100%", height="1100"
        )
        return servidor

    print(f"Abriendo: {url}")
    print("Para detener el servidor, presione Ctrl+C.\n")
    Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
