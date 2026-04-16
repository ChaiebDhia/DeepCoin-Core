import {getRequestConfig} from 'next-intl/server';
import {cookies} from 'next/headers';

export default getRequestConfig(async () => {
  // Read a NEXT_LOCALE cookie to determine language. Default to 'en'
  const cookieStore = await cookies();
  const locale = cookieStore.get('NEXT_LOCALE')?.value || 'en';
  let messages;
  if (locale === 'fr') {
    messages = (await import('../../messages/fr.json')).default;
  } else {
    messages = (await import('../../messages/en.json')).default;
  }

  return {
    locale,
    messages,
    timeZone: 'Europe/Paris'
  };
});

