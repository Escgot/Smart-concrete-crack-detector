import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.crackscan.inspector',
  appName: 'CrackScan',
  webDir: 'frontend',
  bundledWebRuntime: false,
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: "#080a09",
      showSpinner: true,
      androidSpinnerStyle: "large",
      spinnerColor: "#39ff7a",
      splashFullScreen: true,
      splashImmersive: true,
    },
  },
};

export default config;
