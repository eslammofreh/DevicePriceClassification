package com.DevicePriceClassification.DevicePriceClassification.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

@Configuration // Indicates that this is a Spring configuration class
public class AppConfig {

    @Bean // Create a RestTemplate bean for making HTTP requests
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
