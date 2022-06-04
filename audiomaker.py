
from gtts import gTTS


audio=gTTS(text="Lower your arms", lang="en",slow=False,tld="com")
audio.save("lower_arms.mp3")