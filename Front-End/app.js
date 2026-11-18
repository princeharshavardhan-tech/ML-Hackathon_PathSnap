const $ = id => document.getElementById(id);
let apiBase = new URLSearchParams(location.search).get('api') || localStorage.getItem('pathsnapApi') ||
  (location.port === '5173' ? '' : location.origin);

function status(message, kind='ready') {
  $('status-text').textContent = message;
  $('status').className = `status ${kind}`;
}
function percent(value) { return value == null ? '—' : `${(100 * value).toFixed(1)}%`; }
async function request(url, options) {
  if(!apiBase) throw new Error('Enter Laptop 1 API address and click Connect.');
  const response=await fetch(`${apiBase}${url}`,options);
  const body=await response.json();
  if(!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}
async function loadTrips() {
  if(!apiBase){status('Enter Laptop 1 API address','error');return;}
  $('trip-select').disabled=true;$('run-button').disabled=true;
  status('Connecting to Laptop 1…','loading');
  try {
    const [health,trips]=await Promise.all([request('/api/health'),request('/api/trips')]);
    const select=$('trip-select');select.replaceChildren();
    for(const trip of trips.trips){const option=document.createElement('option');option.value=trip;option.textContent=`Trip ${trip}`;select.append(option);}
    select.disabled=!trips.trips.length;
    $('run-button').disabled=!trips.trips.length;
    $('trip-count').textContent=`${trips.trips.length} trips available · ${health.partition.replaceAll('_',' ')}`;
    $('model-name').textContent=health.model.replaceAll('_',' ').replace('.joblib','');
    status('API connected · Ready to run');
  } catch(error){status(error.message,'error');$('trip-count').textContent='Check the address, Wi-Fi, firewall, and API server.';}
}
function connect() {
  try {
    const url=new URL($('api-url').value.trim());
    if(!['http:','https:'].includes(url.protocol)) throw new Error('Use an HTTP or HTTPS address.');
    apiBase=url.origin;
    $('api-url').value=apiBase;
    $('api-docs').href=`${apiBase}/docs`;
    localStorage.setItem('pathsnapApi',apiBase);
    loadTrips();
  } catch(error) {status('Enter a full address such as http://192.168.1.10:8000','error');}
}
async function runTrip() {
  const trip=$('trip-select').value;
  if(!trip) return;
  $('run-button').disabled=true;status(`Processing trip ${trip}…`,'loading');
  try {
    const result=await request(`/api/predict/${encodeURIComponent(trip)}`,{method:'POST'});
    const m=result.metrics;
    $('metric-observations').textContent=m.observations.toLocaleString();
    $('metric-hmm').textContent=percent(m.hmm_accuracy);
    $('metric-classifier').textContent=percent(m.classifier_accuracy);
    $('metric-continuity').textContent=percent(m.route_continuity);
    $('coverage').textContent=m.evaluated_observations == null ? 'Reference route unavailable' : `${m.evaluated_observations.toLocaleString()} high-confidence observations`;
    $('page-title').textContent=`Trip ${trip}`;
    $('map-heading').textContent='Reconstructed route';
    const mapResponse=await fetch(`${apiBase}/api/map/${encodeURIComponent(trip)}`);
    if(!mapResponse.ok){
      const error=await mapResponse.json();
      throw new Error(error.detail || `Map request failed (${mapResponse.status})`);
    }
    $('folium-map').srcdoc=await mapResponse.text();
    $('map-empty').hidden=true;
    status(`Trip ${trip} completed`);
  } catch(error){status(error.message,'error');}
  finally{$('run-button').disabled=false;}
}
$('run-button').addEventListener('click',runTrip);
$('connect-button').addEventListener('click',connect);
$('api-url').addEventListener('keydown',e=>{if(e.key==='Enter')connect();});
if(apiBase){$('api-url').value=apiBase;$('api-docs').href=`${apiBase}/docs`;loadTrips();}
else status('Enter Laptop 1 API address','error');
