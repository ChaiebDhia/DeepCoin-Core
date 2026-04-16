const fs = require('fs');
const fn = 'app/history/[id]/page.tsx';
let txt = fs.readFileSync(fn, 'utf-8');
txt = txt.replace(/const tCoin = useTranslations\(.*?;/g, '');
txt = txt.replace(/(export default function HistoryDetailPage[^{]+{)/, '\n  const tCoin = useTranslations("Coin");');
fs.writeFileSync(fn, txt);