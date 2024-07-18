package com.DevicePriceClassification.DevicePriceClassification.service;

import com.DevicePriceClassification.DevicePriceClassification.model.Device;
import com.DevicePriceClassification.DevicePriceClassification.repository.DeviceRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Service // Indicates that this is a Spring service
public class DeviceService {

    @Autowired // Inject the DeviceRepository
    private DeviceRepository deviceRepository;

    @Autowired // Inject RestTemplate for making HTTP requests
    private RestTemplate restTemplate;

    @Value("${python.api.url}")
    private String pythonApiUrl;

    // Retrieve all devices
    public List<Device> getAllDevices() {
        return deviceRepository.findAll();
    }

    // Retrieve a specific device by ID
    public Device getDeviceById(Long id) {
        return deviceRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Device not found with id: " + id));
    }

    // Add a new device
    public Device addDevice(Device device) {
        return deviceRepository.save(device);
    }

    // Predict price for a device
    @Transactional // Ensure database consistency
    public Device predictPrice(Long deviceId) {
        // Retrieve the device
        Device device = getDeviceById(deviceId);

        // Call Python API to predict price
        ResponseEntity<Integer> response = restTemplate.postForEntity(this.pythonApiUrl, device, Integer.class);
        Integer predictedPrice = response.getBody();

        System.out.println("Predicted Price: " + predictedPrice);
        // Update device with predicted price
        device.setPrice_range(predictedPrice);

        // Save updated device
        return deviceRepository.save(device);
    }
}
