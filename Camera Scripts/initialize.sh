sudo vcgencmd display_power 0
sleep 1
sudo vcgencmd display_power 1
sudo killall fbi
sudo fbi -T 1 -d /dev/fb0 -noverbose worley_pattern.png
