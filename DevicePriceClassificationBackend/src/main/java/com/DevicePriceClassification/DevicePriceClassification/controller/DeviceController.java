package com.DevicePriceClassification.DevicePriceClassification.controller;

import com.DevicePriceClassification.DevicePriceClassification.model.Device;
import com.DevicePriceClassification.DevicePriceClassification.service.DeviceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController // Indicates that this is a Spring REST controller
@RequestMapping("/api/devices") // Base URL for all endpoints in this controller
public class DeviceController {

    @Autowired // Inject the DeviceService
    private DeviceService deviceService;

    // GET endpoint to retrieve all devices
    @GetMapping
    public ResponseEntity<List<Device>> getAllDevices() {
        return ResponseEntity.ok(deviceService.getAllDevices());
    }

    // GET endpoint to retrieve a specific device by ID
    @GetMapping("/{id}")
    public ResponseEntity<Device> getDeviceById(@PathVariable Long id) {
        return ResponseEntity.ok(deviceService.getDeviceById(id));
    }

    // POST endpoint to add a new device
    @PostMapping
    public ResponseEntity<Device> addDevice(@RequestBody Device device) {
        return ResponseEntity.ok(deviceService.addDevice(device));
    }

    // POST endpoint to predict price for a device
    @PostMapping("/predict/{deviceId}")
    public ResponseEntity<Device> predictPrice(@PathVariable Long deviceId) {
        return ResponseEntity.ok(deviceService.predictPrice(deviceId));
    }
}
