import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

export const calculateFspl = async (data) => {
    const response = await axios.post(`${API_BASE_URL}/fspl`, data);
    return response.data;
};

export const calculateCoverage = async (data) => {
    const response = await axios.post(`${API_BASE_URL}/coverage`, data);
    return response.data;
};

export const calculateSrtmProfile = async (data) => {
    const response = await axios.post(`${API_BASE_URL}/srtm/profile`, data);
    return response.data;
};

export const calculatePinn = async (data) => {
    const response = await axios.post(`${API_BASE_URL}/pinn/train_and_infer`, data);
    return response.data;
};

export const calculateCompareProfile = async (data) => {
    const response = await axios.post(`${API_BASE_URL}/pinn/compare_profile`, data);
    return response.data;
};

export const downloadReport = () => {
    window.open(`${API_BASE_URL.replace('/api', '')}/api/report/download`, "_blank");
};

export const getSocialBarrios = async () => {
    const response = await axios.get(`${API_BASE_URL.replace('/api', '')}/api/social/barrios`);
    return response.data;
};

export const getSocialAnalysis = async (data) => {
    const response = await axios.post(`${API_BASE_URL.replace('/api', '')}/api/social/analysis`, data);
    return response.data;
};

export const optimizeStations = async (n_stations = 3) => {
    const response = await axios.post(`${API_BASE_URL.replace('/api', '')}/api/optimizer/stations`, { n_stations, epochs: 50 });
    return response.data;
};

export const getDepartments = async () => {
    const response = await axios.get(`${API_BASE_URL}/stations/departments`);
    return response.data;
};

export const getCities = async (department) => {
    const response = await axios.get(`${API_BASE_URL}/stations/cities/${encodeURIComponent(department)}`);
    return response.data;
};

export const getStations = async (filters = {}) => {
    const params = new URLSearchParams();
    if (filters.dept) params.append('dept', filters.dept);
    if (filters.city) params.append('city', filters.city);
    if (filters.freq) params.append('freq', filters.freq);
    const response = await axios.get(`${API_BASE_URL}/stations?${params.toString()}`);
    return response.data;
};

export const getStation = async (stationId) => {
    const response = await axios.get(`${API_BASE_URL}/stations/${stationId}`);
    return response.data;
};

export const getCochannelInterference = async (data) => {
    const response = await axios.post(`${API_BASE_URL}/interference/cochannel`, data);
    return response.data;
};

export const getSignalAtPoint = async (data) => {
    const response = await axios.post(`${API_BASE_URL}/citizen/signal_at_point`, data);
    return response.data;
};

export const detectIllegal = async (data) => {
    const response = await axios.post(`${API_BASE_URL}/interference/detect_illegal`, data);
    return response.data;
};
