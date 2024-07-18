package com.DevicePriceClassification.DevicePriceClassification.repository;

import com.DevicePriceClassification.DevicePriceClassification.model.Device;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository // Indicates that this is a Spring Data repository
public interface DeviceRepository extends JpaRepository<Device, Long> {
    // JpaRepository provides CRUD operations out of the box.
}